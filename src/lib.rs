use ash::util::read_spv;
pub use ash::vk;
use ash::ext::debug_utils;
use ash::vk::Handle;
use ash::{Entry, Instance, Device};
use ash_window;
use ash::khr::{surface, swapchain};

use electrum_engine_macros::Vertex;
use raw_window_handle::{self, DisplayHandle, WindowHandle};

use anyhow::{anyhow, Result};
use thiserror::Error;
use tracing::*;

use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::{self, CStr};
use std::{fs, ptr};
use std::time::{Duration, Instant};

pub mod command;
pub mod buffer;
pub mod image;
pub mod present;

use command::*;
use buffer::*;
use image::*;
use present::*;

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

/// The required device extensions.
const DEVICE_EXTENSIONS: &[&CStr] = &[ash::khr::swapchain::NAME, ash::khr::dynamic_rendering::NAME];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone)]
pub struct Renderer {
    _entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub data: RendererData,
    pub stats: RenderStats,
    pub width: u32,
    pub height: u32,
    frame: usize,
    pub resized: bool,
}

impl Renderer {
    /// Creates the Vulkan Renderer.
    pub fn new(display_handle: DisplayHandle, window_handle: WindowHandle, width: u32, height: u32) -> Result<Self> {
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

        let image_count = unsafe { create_swapchain(&entry, &instance, &device, &mut data, width, height) }?;
        unsafe { create_swapchain_image_views(&device, &mut data) }?;

        unsafe { create_command_pools(&entry, &instance, &device, &mut data) }.unwrap();
        unsafe { create_command_buffers(&device, &mut data) }.unwrap();

        unsafe { create_sync_objects(&instance, &device, &mut data, image_count) }?;

        let pipeline = create_pipeline(&device, width, height);
        data.pipeline = pipeline;

        let vertices = create_and_stage_buffer(&instance, &device, &data, (std::mem::size_of::<TestVertex>() * 6) as u64, vk::BufferUsageFlags::VERTEX_BUFFER, "Vertex Buffer", &[
            TestVertex{ position: [ 0.0 , -0.5 ], colour: [0.0 , 1.0 , 0.0] },
            TestVertex{ position: [-0.25, -0.25], colour: [0.25, 0.75, 0.25] },
            TestVertex{ position: [ 0.25, -0.25], colour: [0.25, 0.75, 0.25] },
            TestVertex{ position: [-0.25,  0.25], colour: [0.75, 0.25, 0.75] },
            TestVertex{ position: [ 0.25, -0.25], colour: [0.75, 0.25, 0.75] },
            TestVertex{ position: [ 0.0 ,  0.5 ], colour: [1.0 , 0.0 , 1.0] },
        ]).unwrap();
        data.vertices = vertices;

        let indices = create_and_stage_buffer(&instance, &device, &data, (std::mem::size_of::<u32>() * 12) as u64, vk::BufferUsageFlags::INDEX_BUFFER, "Index Buffer", &[
            0, 1, 2,
            1, 2, 4,
            1, 4, 3,
            3, 4, 5,
        ]).unwrap();
        data.indices = indices;

        let colour_attachement = Image::new(
            &instance,
            &device,
            &data,
            width,
            height,
            MipLevels::One,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageViewType::TYPE_2D,
            1,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::ImageAspectFlags::COLOR,
            false
        );
        data.colour_attachment = colour_attachement;


        let stats = RenderStats {
            start: Instant::now(),
            delta: Duration::ZERO,
            delta_start: Instant::now(),
            frame: 0,
        };

        info!("finished initialising");

        Ok(Renderer {
            _entry: entry,
            instance,
            data,
            device,
            frame: 0,
            width,
            height,
            resized: false,
            stats,
        })
    }

    /// This function renders a frame
    ///
    /// # Safety
    /// Do not call this function if the window is minimised or if `destroy` has been called
    pub unsafe fn render(&mut self, width: u32, height: u32) -> Result<()> {
        let render_fence = self.data.render_fences[self.frame];

        self.device.wait_for_fences(&[render_fence], true, 1000000000)?;
        self.device.reset_fences(&[render_fence])?;

        let swapchain_loader = ash::khr::swapchain::Device::new(&self.instance, &self.device);
        let result = swapchain_loader.acquire_next_image(
            self.data.swapchain,
            1000000000,
            self.data.swapchain_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                warn!("acquire: Out of date");
                return self.recreate_swapchain(width, height)
            },
            Err(e) => {
                warn!("acquire: error");
                return Err(anyhow!(e))
            },
        };

        self.update_command_buffer(image_index)?;

        
        let command_info = [command_buffer_submit_info(&self.data.command_buffers[image_index])];
        let wait_semaphore_info = [semaphore_submit_info(vk::PipelineStageFlags2::BOTTOM_OF_PIPE, &self.data.swapchain_semaphores[self.frame])];
        let signal_semaphore_info = [semaphore_submit_info(vk::PipelineStageFlags2::ALL_GRAPHICS, &self.data.render_semaphores[image_index])];
        
        let submit_info = submit_info(&command_info, &signal_semaphore_info, &wait_semaphore_info);
        
        self.device.queue_submit2(self.data.graphics_queue, &[submit_info], render_fence)?;

        let wait_semaphores = &[self.data.render_semaphores[image_index]];
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = swapchain_loader.queue_present(self.data.present_queue, &present_info);
        let changed = 
            result == Ok(true)
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


        self.stats.delta = self.stats.delta_start.elapsed();
        self.stats.delta_start = Instant::now();

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        let command_pool = self.data.command_pools[image_index];
        self.device
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let attachment_infos = [
            vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.05, 0.5, 0.0, 1.0]}})
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image_view(self.data.colour_attachment.view)
        ];

        let rendering_info = vk::RenderingInfo::default()
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: self.width, height: self.height },
            })
            .color_attachments(&attachment_infos);

        self.device.begin_command_buffer(command_buffer, &info)?;

        transition_image(&self.device, command_buffer, self.data.colour_attachment.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        self.device.cmd_begin_rendering(command_buffer, &rendering_info);

        self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.data.pipeline);
        self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.vertices.buffer], &[0]);
        self.device.cmd_bind_index_buffer(command_buffer, self.data.indices.buffer, 0, vk::IndexType::UINT32);
        self.device.cmd_draw_indexed(command_buffer, 12, 1, 0, 0, 0);

        self.device.cmd_end_rendering(command_buffer);

        transition_image(&self.device, command_buffer, self.data.colour_attachment.image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        transition_image(&self.device, command_buffer, self.data.swapchain_images[image_index], vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        copy_image_to_image(&self.device, command_buffer, self.data.colour_attachment.image, self.data.swapchain_images[image_index], vk::Extent2D { width: self.width, height: self.height }, vk::Extent2D { width: self.width, height: self.height });

        transition_image(&self.device, command_buffer, self.data.swapchain_images[image_index], vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);

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

        create_command_buffers(&self.device, &mut self.data)?;

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
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
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.device.destroy_pipeline(self.data.pipeline, None);
        
        let swapchain_loader = swapchain::Device::new(&self.instance, &self.device);
        swapchain_loader.destroy_swapchain(self.data.swapchain, None);
        self.data
            .command_pools
            .iter()
            .for_each(|p| self.device.destroy_command_pool(*p, None));

        self.device
            .destroy_command_pool(self.data.command_pool, None);

        self.data
            .render_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));

        self.data
            .render_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));

        self.data
            .swapchain_semaphores
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

    // Command Buffers
    pub command_pool: vk::CommandPool,
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,

    // Semaphores
    pub swapchain_semaphores: Vec<vk::Semaphore>,
    pub render_semaphores: Vec<vk::Semaphore>,

    // Fences
    pub render_fences: Vec<vk::Fence>,

    // MSAA
    pub msaa_samples: vk::SampleCountFlags,

    // Temp
    vertices: BufferWrapper,
    indices: BufferWrapper,
    pipeline: vk::Pipeline,

    colour_attachment: Image,

    pub recreated: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct RenderStats {
    start: Instant,

    delta: Duration,
    delta_start: Instant,

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

    let mut features1_2 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .descriptor_indexing(true);

    let mut features1_3 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);

    // Create

    let info = vk::DeviceCreateInfo::default()
        .push_next(&mut features1_2)
        .push_next(&mut features1_3)
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
    image_count: u32,
) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    for i in 0..image_count {
        let swapchain_semaphore = device.create_semaphore(&semaphore_info, None)?;
        set_object_name(
            instance,
            device,
            &format!("Image Available Semaphore {}", i),
            swapchain_semaphore,
        )?;
        data.swapchain_semaphores
            .push(swapchain_semaphore);

        let render_semaphore = device.create_semaphore(&semaphore_info, None)?;
        set_object_name(
            instance,
            device,
            &format!("Render Finished Semaphore {}", i),
            render_semaphore,
        )?;
        data.render_semaphores
            .push(render_semaphore);

        let render_fence = device.create_fence(&fence_info, None)?;
        set_object_name(
            instance,
            device,
            &format!("In Flight Fence {}", i),
            render_fence,
        )?;
        data.render_fences.push(render_fence);
    }

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


// --------------------- //
// ----- Debugging ----- //
// --------------------- //

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


fn create_pipeline(device: &Device, width: u32, height: u32) -> vk::Pipeline {
    let bytecode = read_spv(&mut fs::File::open("res/shaders/test.vert.spv").unwrap()).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::default()
        .code(&bytecode);

    let vertex = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

    let vertex_stage_info = vk::PipelineShaderStageCreateInfo::default()
        .module(vertex)
        .name(c"main")
        .stage(vk::ShaderStageFlags::VERTEX);


    let bytecode = read_spv(&mut fs::File::open("res/shaders/test.frag.spv").unwrap()).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::default()
        .code(&bytecode);

    let fragment = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

    let fragment_stage_info = vk::PipelineShaderStageCreateInfo::default()
        .module(fragment)
        .name(c"main")
        .stage(vk::ShaderStageFlags::FRAGMENT);

    let attributs = TestVertex::attribute_descriptions();
    let bindings = TestVertex::binding_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&attributs)
        .vertex_binding_descriptions(&bindings);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let attachments = [
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
    ];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&attachments)
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .flags(vk::PipelineMultisampleStateCreateFlags::empty())
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        }];
    
    let viewports = [vk::Viewport::default()
        .width(width as f32)
        .height(height as f32)
        .min_depth(0.01)
        .max_depth(1.0)
        .x(0.0)
        .y(0.0)];

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissors(&scissors)
        .viewports(&viewports);


    let layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .flags(vk::PipelineLayoutCreateFlags::empty())
        .set_layouts(&[])
        .push_constant_ranges(&[]);

    let layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }.unwrap();


    let mut pipeline_info2 = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&[vk::Format::R16G16B16A16_SFLOAT]);

    let binding = [vertex_stage_info, fragment_stage_info];
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .color_blend_state(&color_blend_state)
        .multisample_state(&multisample_state)
        .rasterization_state(&rasterization_state)
        .viewport_state(&viewport_state)
        .layout(layout)
        .push_next(&mut pipeline_info2)
        .stages(&binding);

    unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) }.unwrap()[0]
}

pub trait Vertex {
    fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription>;
    fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>;
}

#[repr(C)]
#[derive(Debug, Vertex)]
struct TestVertex {
    #[vertex(format = "R32G32_SFLOAT")]
    position: [f32; 2],
    #[vertex(format = "R32G32B32_SFLOAT")]
    colour: [f32; 3]
}
