pub use ash::vk;
use ash::ext::debug_utils;
use ash::vk::{Extent2D, Handle};
use ash::{Entry, Instance, Device};
use ash_window;
use ash::khr::{surface, swapchain};

use raw_window_handle::{self, DisplayHandle, WindowHandle};

use anyhow::{anyhow, Result};
use thiserror::Error;
use tracing::*;

use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::{self, CStr};
use std::ptr;
use std::time::{Duration, Instant};

mod buffer;
mod camera;
mod command;
mod light;
mod model;
mod present;
mod render_pass;
mod shader;
mod texture;
mod descriptor;

pub use buffer::*;
pub use camera::*;
pub use light::*;
pub use mesh::*;
pub use model::*;
pub use present::*;
pub use render_pass::*;
pub use shader::*;
pub use texture::*;
pub use record::*;
pub use descriptor::*;

use command::{create_command_buffers, create_command_pools};

pub use electrum_engine_macros;

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

/// The required device extensions.
const DEVICE_EXTENSIONS: &[&CStr] = &[ash::khr::swapchain::NAME];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone)]
pub struct Renderer {
    _entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub data: RendererData,
    pub stats: RenderStats,
    frame: usize,
    pub resized: bool,
}

impl Renderer {
    /// Creates the Vulkan Renderer.
    pub fn create(display_handle: DisplayHandle, window_handle: WindowHandle, width: u32, height: u32) -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        let mut data = RendererData::default();

        let instance = unsafe { create_instance(display_handle.into(), &entry, &mut data)? };
        data.surface = unsafe { ash_window::create_surface(
            &entry,
            &instance,
            display_handle.into(),
            window_handle.into(),
            None,
        )}.unwrap();

        unsafe { pick_physical_device(&entry, &instance, &mut data)? };
        let device = unsafe { create_logical_device(&entry, &instance, &mut data)? };

        unsafe { create_swapchain(&entry, &instance, &device, &mut data, width, height) }?;
        unsafe { create_swapchain_image_views(&device, &mut data) }?;

        unsafe { create_command_pools(&entry, &instance, &device, &mut data) }.unwrap();
        unsafe { create_command_buffers(&device, &mut data) }.unwrap();

        unsafe { create_sync_objects(&instance, &device, &mut data) }?;

        let stats = RenderStats {
            start: Instant::now(),
            delta: Duration::ZERO,
            delda_start: Instant::now(),
            frame: 0,
        };

        Ok(Renderer {
            _entry: entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            stats,
        })
    }

    /// This function renders a frame
    ///
    /// # Safety
    /// Do not call this function if the window is minimised or if `destroy` has been called
    pub unsafe fn render(&mut self, width: u32, height: u32) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let swapchain_loader = ash::khr::swapchain::Device::new(&self.instance, &self.device);

        let result = swapchain_loader.acquire_next_image(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(width, height),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device
                .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        let mut render_passes = self.data.render_passes.clone();
        for render_pass in render_passes.iter_mut() {
            for subpass in render_pass.subpasses.iter_mut() {
                subpass.update(&self.device, &mut self.data, &self.stats, image_index);
            }
        }
        self.data.render_passes = render_passes;

        self.update_command_buffer(image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[in_flight_fence])?;

        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = swapchain_loader.queue_present(self.data.present_queue, &present_info);
        let changed = result == Ok(true)
            || result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR);

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(width, height)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        } else {
            self.data.recreated = false;
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        self.stats.frame += 1;

        self.stats.delta = self.stats.delda_start.elapsed();
        self.stats.delda_start = Instant::now();

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        let command_pool = self.data.command_pools[image_index];
        self.device
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE);

        self.device.begin_command_buffer(command_buffer, &info)?;

        
        let mut render_passes = self.data.render_passes.clone();
        for render_pass in render_passes.iter_mut() {

            let render_area = vk::Rect2D::default()
                .offset(vk::Offset2D::default())
                .extent(Extent2D {
                    width: render_pass.width,
                    height: render_pass.height,
                });

            let info = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass.render_pass)
                .framebuffer(render_pass.framebuffers[image_index])
                .render_area(render_area)
                .clear_values(&render_pass.clear_values);

            begin_command_label(
                &self.instance,
                &self.device,
                command_buffer,
                &render_pass.name,
                [0.0, 0.1, 0.5, 1.0],
            );

            self.device.cmd_begin_render_pass(
                command_buffer,
                &info,
                vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
            );

            render_pass.subpasses.iter_mut().for_each(|s| {
                s.record_command_buffers(&self.instance, &self.device, &self.data, render_pass.render_pass, render_pass.framebuffers[image_index], image_index)
                    .unwrap()
            });

            for (i, render_data) in render_pass.subpasses.iter().enumerate() {
                if i > 0 {
                    self.device.cmd_next_subpass(
                        command_buffer,
                        vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
                    );
                }

                let secondary_command_buffers = vec![render_data.command_buffers[image_index]];

                begin_command_label(&self.instance, &self.device, command_buffer, "Draw Calls", [0.0, 1.0, 1.0, 1.0]);
                self.device
                    .cmd_execute_commands(command_buffer, &secondary_command_buffers);
                end_command_label(&self.instance, &self.device, command_buffer);
            }

            self.device.cmd_end_render_pass(command_buffer);

            end_command_label(&self.instance, &self.device, command_buffer);   
        }
        self.data.render_passes = render_passes;

        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        self.data.recreated = true;

        info!("Recreating swapchain");

        self.device.device_wait_idle()?;
        self.destroy_swapchain();

        let old_swapchain = self.data.swapchain;
        create_swapchain(&self._entry, &self.instance, &self.device, &mut self.data, width, height)?;

        let swapchain_loader = swapchain::Device::new(&self.instance, &self.device);
        swapchain_loader.destroy_swapchain(old_swapchain, None);

        create_swapchain_image_views(&self.device, &mut self.data)?;

        let mut render_passes = self.data.render_passes.clone();
        for render_pass in render_passes.iter_mut() {
            render_pass.recreate_swapchain(&self.instance, &self.device, &mut self.data)?;
        }
        self.data.render_passes = render_passes;

        let mut objects = self.data.objects.clone();

        objects
            .iter_mut()
            .for_each(|o| o.recreate_swapchain(&self.device, &mut self.data));

        self.data.objects = objects;

        create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        let render_passes = self.data.render_passes.clone();
        render_passes.iter().for_each(|r| r.destroy_swapchain(&self.device, &mut self.data));
        self.data.render_passes = render_passes;

        self.data
            .objects
            .iter()
            .for_each(|o| o.destroy_swapchain(&self.device));

        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
    }

    /// This function will destroy all vulkan objects in the renderer
    ///
    /// # Safety
    /// This function **MUST** be called befoore the end of the program
    /// and **MUST** be the last function called on the renderer
    unsafe fn destroy(&mut self) {
        self.destroy_swapchain();
        
        let swapchain_loader = swapchain::Device::new(&self.instance, &self.device);
        swapchain_loader.destroy_swapchain(self.data.swapchain, None);
        self.data
            .command_pools
            .iter()
            .for_each(|p| self.device.destroy_command_pool(*p, None));

        self.device
            .destroy_command_pool(self.data.command_pool, None);

        let mut objects = self.data.objects.clone();
        objects
            .iter_mut()
            .for_each(|o| o.destroy(&self.device));
        self.data.objects = objects;

        self.data
            .shaders
            .iter()
            .for_each(|s| s.destroy(&self.device));

        self.data
            .textures
            .iter()
            .for_each(|t| t.destroy(&self.device));

        self.data
            .cameras
            .iter()
            .for_each(|c| c.destroy(&self.device));

        self.data.other_descriptors.iter().for_each(|d| d.destroy(&self.device));

        self.data.global_descriptor_pools.destroy(&self.device);
        self.data.global_layout_cache.destroy(&self.device);

        self.data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));

        self.data
            .render_finished_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));

        self.data
            .image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));

        self.device.destroy_device(None);

        let surface_loader = surface::Instance::new(&self._entry, &self.instance);
        surface_loader.destroy_surface(self.data.surface, None);

        if VALIDATION_ENABLED {
            let debug_utils_loader = debug_utils::Instance::new(&self._entry, &self.instance);
            debug_utils_loader.destroy_debug_utils_messenger(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe { self.destroy() };
    }
}

/// The Vulkan handles and associated properties used by the Vulkan app.
#[derive(Default, Clone)]
pub struct RendererData {
    // Debug
    pub messenger: vk::DebugUtilsMessengerEXT,

    // Surface
    pub surface: vk::SurfaceKHR,

    // Physical Device / Logical Device
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,

    // Swapchain
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,

    // Pipeline
    pub render_passes: Vec<RenderPass>,

    // Command Buffers
    pub command_pool: vk::CommandPool,
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,

    // objects
    pub shaders: Record<Box<dyn Shader>>,
    pub textures: Record<Image>,
    pub point_light_data: Record<PointLight>,

    pub objects: Record<Mesh>,
    pub materials: Record<Box<dyn Material>>,
    pub cameras: Record<Box<dyn Camera>>,
    pub other_descriptors: Record<Box<dyn DescriptorSet>>,

    pub global_descriptor_pools: DescriptorAllocator,
    pub global_layout_cache: DescriptorLayoutCache,

    // Semaphores
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,

    // Fences
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,

    // MSAA
    pub msaa_samples: vk::SampleCountFlags,

    pub recreated: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct RenderStats {
    start: Instant,

    delta: Duration,
    delda_start: Instant,

    frame: u64,
}

pub fn align_string(string: &str) -> [i8; 256] {
    let mut aligned = [0; 256];
    for (i, c) in string.chars().enumerate() {
        aligned[i] = c as i8;
    }
    aligned
}

unsafe fn create_instance(
    display_handle: raw_window_handle::RawDisplayHandle,
    entry: &Entry,
    data: &mut RendererData,
) -> Result<Instance> {
    // Application Info

    let application_info = vk::ApplicationInfo::default()
        .application_name(c"Electrum Engine")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"Electrum Engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    // Layers

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&align_string("VK_LAYER_KHRONOS_validation")) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layer_names = if VALIDATION_ENABLED {
        vec![c"VK_LAYER_KHRONOS_validation"]
    } else {
        Vec::new()
    };

    let layers = layer_names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();

    // Extensions
    let mut extensions = ash_window::enumerate_required_extensions(display_handle)
        .unwrap()
        .to_vec();

    // Required by Vulkan SDK on macOS since 1.3.216.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
        extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
    }

    let flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    if VALIDATION_ENABLED {
        extensions.push(debug_utils::NAME.as_ptr());
    }

    // Create

    let mut info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    // Messenger

    if VALIDATION_ENABLED {
        let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
        data.messenger = debug_utils_loader.create_debug_utils_messenger(&debug_info, None)?;
    }

    Ok(instance)
}

#[derive(Debug, Error)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);

unsafe fn pick_physical_device(entry: &Entry, instance: &Instance, data: &mut RendererData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(entry, instance, data, physical_device) {
            warn!(
                "Skipping physical device: {}: {}",
                String::from_utf8(properties.device_name.iter().map(|b| *b as u8).collect::<_>())?, error
            );
        } else {
            info!("Selected physical device: {}.", String::from_utf8(properties.device_name.iter().map(|b| *b as u8).collect::<_>())?);
            data.physical_device = physical_device;
            data.msaa_samples = get_max_msaa_samples(instance, data);

            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn check_physical_device(
    entry: &Entry,
    instance: &Instance,
    data: &RendererData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(entry, instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(entry, instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
    }

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(&align_string(e.to_str().unwrap()))) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}

unsafe fn get_max_msaa_samples(instance: &Instance, data: &RendererData) -> vk::SampleCountFlags {
    let properties = instance.get_physical_device_properties(data.physical_device);
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;
    [
        vk::SampleCountFlags::TYPE_64,
        vk::SampleCountFlags::TYPE_32,
        vk::SampleCountFlags::TYPE_16,
        vk::SampleCountFlags::TYPE_8,
        vk::SampleCountFlags::TYPE_4,
        vk::SampleCountFlags::TYPE_2,
    ]
    .iter()
    .cloned()
    .find(|c| counts.contains(*c))
    .unwrap_or(vk::SampleCountFlags::TYPE_1)
}

unsafe fn create_logical_device(entry: &Entry, instance: &Instance, data: &mut RendererData) -> Result<Device> {
    // Queue Create Infos

    let indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Extensions

    let extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Features

    // change this for other features
    let features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(true)
        .fill_mode_non_solid(true)
        .wide_lines(true);

    // Create

    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    // Queues
    let graphics_queue = device.get_device_queue(indices.graphics, 0);
    set_object_name(
        instance,
        &device,
        "Graphics Queue",
        graphics_queue,
    )?;
    data.graphics_queue = graphics_queue;

    let present_queue = device.get_device_queue(indices.present, 0);
    set_object_name(
        instance,
        &device,
        "Present Queue",
        graphics_queue,
    )?;
    data.present_queue = present_queue;

    Ok(device)
}

pub fn get_depth_format(instance: &Instance, data: &RendererData) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    unsafe { 
        get_supported_format(
            instance,
            data,
            candidates,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        ) 
    }
}

unsafe fn get_supported_format(
    instance: &Instance,
    data: &RendererData,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f| {
            let properties =
                instance.get_physical_device_format_properties(data.physical_device, *f);

            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find supported format!"))
}

unsafe fn create_sync_objects(
    instance: &Instance,
    device: &Device,
    data: &mut RendererData,
) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    for i in 0..MAX_FRAMES_IN_FLIGHT {
        let image_available_semaphor = device.create_semaphore(&semaphore_info, None)?;
        set_object_name(
            instance,
            device,
            &format!("Image Available Semaphore {}", i),
            image_available_semaphor,
        )?;
        data.image_available_semaphores
            .push(image_available_semaphor);

        let render_finished_semaphor = device.create_semaphore(&semaphore_info, None)?;
        set_object_name(
            instance,
            device,
            &format!("Render Finished Semaphore {}", i),
            render_finished_semaphor,
        )?;
        data.render_finished_semaphores
            .push(render_finished_semaphor);

        let in_flight_fence = device.create_fence(&fence_info, None)?;
        set_object_name(
            instance,
            device,
            &format!("In Flight Fence {}", i),
            in_flight_fence,
        )?;
        data.in_flight_fences.push(in_flight_fence);
    }

    data.images_in_flight = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        data: &RendererData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let surface_loader = surface::Instance::new(&entry, &instance);

        let mut present = None;
        for (index, _) in properties.iter().enumerate() {
            if surface_loader.get_physical_device_surface_support(
                physical_device,
                index as u32,
                data.surface,
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        data: &RendererData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {

        let surface_loader = surface::Instance::new(&entry, &instance);

        Ok(Self {
            capabilities: surface_loader
                .get_physical_device_surface_capabilities(physical_device, data.surface)?,
            formats: surface_loader
                .get_physical_device_surface_formats(physical_device, data.surface)?,
            present_modes: surface_loader
                .get_physical_device_surface_present_modes(physical_device, data.surface)?,
        })
    }
}

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    if message_severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!(
            "{}",
            message
        );
    } else if message_severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!(
            "{}",
            message
        );
    } else if message_severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!(
            "{}",
            message
        );
    } else {
        trace!(
            "{}",
            message
        );
    }

    vk::FALSE
}

fn set_object_name(
    instance: &Instance,
    device: &Device,
    name: &str,
    object_handle: impl Handle,
) -> Result<()> {
    if VALIDATION_ENABLED {
        let name_string = name.to_owned() + "\0";
        let name_cstr = unsafe { CStr::from_bytes_with_nul_unchecked(name_string.as_bytes()) };
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_name(name_cstr)
            .object_handle(object_handle);

        let debug_device = debug_utils::Device::new(instance, device);
        unsafe { debug_device.set_debug_utils_object_name(&name_info) }?
    }

    Ok(())
}

fn begin_command_label(
    instance: &Instance,
    device: &Device,
    command_buffer: vk::CommandBuffer,
    name: &str,
    colour: [f32; 4],
) {
    if VALIDATION_ENABLED {
        let name_string = name.to_owned() + "\0";
        let name_cstr = unsafe { CStr::from_bytes_with_nul_unchecked(name_string.as_bytes()) };
        let info = vk::DebugUtilsLabelEXT::default()
            .label_name(name_cstr)
            .color(colour);

        let debug_device = debug_utils::Device::new(instance, device);
        unsafe { debug_device.cmd_begin_debug_utils_label(command_buffer, &info) }
    }
}

fn end_command_label(instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer) {
    if VALIDATION_ENABLED {
        let debug_device = debug_utils::Device::new(instance, device);
        unsafe { debug_device.cmd_end_debug_utils_label(command_buffer) }
    }
}

/// This will place a label for debbuging applications to display at that point
/// If all the values in the `colour` array are 0.0 the colour wil be ignored
fn insert_command_label(
    instance: &Instance,
    device: &Device,
    command_buffer: vk::CommandBuffer,
    name: &str,
    colour: [f32; 4],
) {
    if VALIDATION_ENABLED {
        let name_string = name.to_owned() + "\0";
        let name_cstr = unsafe { CStr::from_bytes_with_nul_unchecked(name_string.as_bytes()) };
        let info = vk::DebugUtilsLabelEXT::default()
            .label_name(name_cstr)
            .color(colour);

        let debug_device = debug_utils::Device::new(instance, device);

        unsafe { debug_device.cmd_insert_debug_utils_label(command_buffer, &info) };
    }
}

pub fn get_c_ptr_slice<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        ptr::null()
    } else {
        slice.as_ptr()
    }
}

pub fn get_c_ptr<T>(t: &T) -> *const T
where
    T: Default,
    T: Eq,
{
    if t == &T::default() {
        ptr::null()
    } else {
        t
    }
}

mod record {
    use std::{ops::{Index, IndexMut}, slice::{Iter, IterMut}, vec::IntoIter};

    use thiserror::Error;

    pub trait Loadable {
        fn is_loaded(&self) -> bool;
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Record<T>(Vec<T>);

    impl<T> Record<T>
        where T: Loadable
    {
        pub fn new() -> Self {
            Record(vec![])
        }

        pub fn from_vec(vec: Vec<T>) -> Self {
            Record(vec)
        }

        pub fn get_loaded(&self, index: usize) -> Result<&T, RecordGetError> {
            match self.0.get(index) {
                Some(value) => {
                    match value.is_loaded() {
                        true => Ok(value),
                        false => Err(RecordGetError::NotLoaded(index)),
                    }
                },
                None => Err(RecordGetError::NotInRecord(index)),
            }
        }

        pub fn get_mut_loaded(&mut self, index: usize) -> Result<&mut T, RecordGetError> {
            match self.0.get_mut(index) {
                Some(value) => {
                    match value.is_loaded() {
                        true => Ok(value),
                        false => Err(RecordGetError::NotLoaded(index)),
                    }
                },
                None => Err(RecordGetError::NotInRecord(index)),
            }
        }

        pub fn get(&self, index: usize) -> Result<&T, RecordGetError> {
            match self.0.get(index) {
                Some(value) => Ok(value),
                None => Err(RecordGetError::NotInRecord(index)),
            }
        }

        pub fn get_mut(&mut self, index: usize) -> Result<&mut T, RecordGetError> {
            match self.0.get_mut(index) {
                Some(value) => Ok(value),
                None => Err(RecordGetError::NotInRecord(index)),
            }
        }

        pub fn push(&mut self, value: T) -> usize {
            let index = self.0.len();
            self.0.push(value);
            index
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }

        pub fn iter(&self) -> Iter<T> {
            self.0.iter()
        }

        pub fn iter_mut(&mut self) -> IterMut<T> {
            self.0.iter_mut()
        }
    }

    impl<T> Index<usize> for Record<T> 
    where T: Loadable
    {
        type Output = T;
    
        fn index(&self, index: usize) -> &Self::Output {
            self.get_loaded(index).unwrap()
        }
    }

    impl<T> IndexMut<usize> for Record<T>
    where T: Loadable
    {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            self.get_mut_loaded(index).unwrap()
        }
    }

    impl<T> Default for Record<T>
    where T: Loadable
    {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T> IntoIterator for Record<T> {
        type Item = T;
    
        type IntoIter = IntoIter<T>;
    
        fn into_iter(self) -> Self::IntoIter {
            self.0.into_iter()
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
    pub enum RecordGetError {
        #[error("Element with index {0} is not in record")]
        NotInRecord(usize),
        #[error("Element with index {0} is not loaded")]
        NotLoaded(usize),
    }

    #[cfg(test)]
    mod tests {
        use crate::record::RecordGetError;

        use super::{Loadable, Record};

        #[derive(Debug, PartialEq)]
        struct Value(bool);

        impl Loadable for Value {
            fn is_loaded(&self) -> bool {
                self.0
            }
        }

        #[test]
        fn empty() {
            let record: Record<Value> = Record::new();
            assert_eq!(record, Record::<Value>(vec![]))
        }

        #[test]
        fn push_multiple() {
            let mut record: Record<Value> = Record::new();
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));

            assert_eq!(record, Record::<Value>(vec![
                Value(true),
                Value(false),
                Value(true),
                Value(false),
                Value(true),
            ]));
        }

        #[test]
        fn get() {
            let mut record: Record<Value> = Record::new();
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));

            assert_eq!(record.get_loaded(0), Result::Ok(&Value(true)));
            assert_eq!(record.get_loaded(5), Result::Err(RecordGetError::NotInRecord(5)));
            assert_eq!(record.get_loaded(1), Result::Err(RecordGetError::NotLoaded(1)));
        }

        #[test]
        fn get_mut() {
            let mut record: Record<Value> = Record::new();
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));

            assert_eq!(record.get_mut_loaded(0), Result::Ok(&mut Value(true)));
            assert_eq!(record.get_mut_loaded(5), Result::Err(RecordGetError::NotInRecord(5)));
            assert_eq!(record.get_mut_loaded(1), Result::Err(RecordGetError::NotLoaded(1)));
        }

        #[test]
        fn get_no_check() {
            let mut record: Record<Value> = Record::new();
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));

            assert_eq!(record.get(0), Result::Ok(&Value(true)));
            assert_eq!(record.get(5), Result::Err(RecordGetError::NotInRecord(5)));
            assert_eq!(record.get(1), Result::Ok(&Value(false)));
        }

        #[test]
        fn get_mut_no_check() {
            let mut record: Record<Value> = Record::new();
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));
            record.push(Value(false));
            record.push(Value(true));

            assert_eq!(record.get_mut(0), Result::Ok(&mut Value(true)));
            assert_eq!(record.get_mut(5), Result::Err(RecordGetError::NotInRecord(5)));
            assert_eq!(record.get_mut(1), Result::Ok(&mut Value(false)));
        }
    }
}
