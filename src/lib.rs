use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
use vulkanalia::vk::{AttachmentDescription, ExtDebugUtilsExtension};
use vulkanalia::window as vk_window;

use anyhow::{anyhow, Result};
use glam::{vec3, Mat4, Quat, Vec3};
use thiserror::Error;
use tracing::*;
use winit::window::Window;

use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;
use std::ffi::CStr;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::os::raw::c_void;
use std::time::{Duration, Instant};

mod buffer;
mod camera;
mod command;
mod light;
mod model;
mod present;
mod shader;
mod texture;

use buffer::*;
use camera::*;
use command::*;
use light::*;
use mesh::*;
use model::*;
use present::*;
use shader::*;
use texture::*;

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

/// The name of the validation layers.
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// The required device extensions.
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone)]
pub struct Renderer {
    _entry: Entry,
    instance: Instance,
    pub device: Device,
    data: RendererData,
    stats: RenderStats,
    frame: usize,
    pub resized: bool,
}

impl Renderer {
    /// Creates the Vulkan Renderer.
    pub fn create(window: &Window) -> Result<Self> {
        let loader = unsafe { LibloadingLoader::new(LIBRARY)? };
        let entry = unsafe { Entry::new(loader) }.map_err(|b| anyhow!("{}", b))?;

        let mut data = RendererData::default();

        let instance = unsafe { create_instance(window, &entry, &mut data)? };
        data.surface = unsafe { vk_window::create_surface(&instance, &window, &window)? };

        unsafe { pick_physical_device(&instance, &mut data)? };
        let device = unsafe { create_logical_device(&instance, &mut data)? };

        unsafe { create_swapchain(window, &instance, &device, &mut data) }?;
        unsafe { create_swapchain_image_views(&device, &mut data) }?;

        data.attachments = vec![
            vk::AttachmentDescription::builder()
                .format(data.swapchain_format)
                .samples(data.msaa_samples)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build(),
            vk::AttachmentDescription::builder()
                .format(unsafe { get_depth_format(&instance, &data) }?)
                .samples(data.msaa_samples)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
            vk::AttachmentDescription::builder()
                .format(data.swapchain_format)
                .samples(vk::SampleCountFlags::_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .build(),
        ];

        data.subpass_data = vec![SubpassData {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            attachments: vec![
                (
                    0,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    AttachmentType::Color,
                ),
                (
                    1,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    AttachmentType::DepthStencil,
                ),
                (
                    2,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    AttachmentType::Resolve,
                ),
            ],
            dependencies: vec![
                vk::SubpassDependency::builder()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .build(),
                vk::SubpassDependency::builder()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                    .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                    .build(),
            ],
        }];

        data.subpass_number = data.subpass_data.len();
        data.render_pass =
            generate_render_pass(&mut data.subpass_data, &data.attachments, &device)?;

        let mut image_data = data
            .attachments
            .iter()
            .map(|a| {
                let (usage, aspect) = match a.final_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        vk::ImageAspectFlags::COLOR,
                    ),
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        vk::ImageAspectFlags::DEPTH,
                    ),
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        vk::ImageAspectFlags::DEPTH,
                    ),
                    vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL => (
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        vk::ImageAspectFlags::STENCIL,
                    ),
                    _ => (
                        vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        vk::ImageAspectFlags::COLOR,
                    ),
                };

                (*a, usage, aspect)
            })
            .collect::<Vec<_>>();

        image_data
            .iter_mut()
            .for_each(|d| *d = (d.0, d.1 | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT, d.2));

        data.images = generate_render_pass_images(image_data, &data, &instance, &device);
        data.framebuffers = unsafe { create_framebuffers(&data, &device) }.unwrap();

        unsafe { create_command_pools(&instance, &device, &mut data) }.unwrap();
        unsafe { create_command_buffers(&device, &mut data) }.unwrap();

        unsafe { create_sync_objects(&instance, &device, &mut data) }?;

        let aspect = data.swapchain_extent.width as f32 / data.swapchain_extent.height as f32;

        let projection = Projection::new(PI / 4.0, aspect, 0.1, 100.0);
        let mut camera = SimpleCamera::new(
            &instance,
            &device,
            &data,
            vec3(0.0, 4.0, 4.0),
            vec3(0.0, 0.0, 0.0),
            projection,
        );
        camera.look_at(vec3(0.0, 0.0, 0.0), Vec3::NEG_Y);

        data.camera = Box::new(camera.clone());

        let view = data.camera.view();
        let proj = data.camera.proj();

        let light = PointLight::new(vec3(3.0, 3.0, 0.0), vec3(1.0, 1.0, 1.0), 5.0);
        let light_buffers = light.get_buffers(&instance, &device, &data)?;

        data.point_lights.push(light_buffers.clone());

        let position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

        let lit_shader = VFShader::builder(&instance, &device, "Lit".to_string())
            .compile_vertex("res\\shaders\\vert.glsl")
            .compile_fragment("res\\shaders\\frag.glsl")
            .build();

        let lit_shader_hash = lit_shader.hash();

        data.shaders.insert(lit_shader_hash, Box::new(lit_shader));

        let image = Image::from_path(
            "res\\textures\\white.png",
            MipLevels::Maximum,
            &instance,
            &device,
            &data,
            vk::Format::R8G8B8A8_SRGB,
        );
        
        let sampler = unsafe {
            create_texture_sampler(
                &instance,
                &device,
                &image.mip_level,
                "white sampler".to_string(),
            )
        }
        .unwrap();

        let sampler_hash = get_hash(&sampler);
        data.samplers.insert(sampler_hash, sampler);

        let image_hash = get_hash(&image);
        data.textures.insert(image_hash, image);

        let mut monkey = ObjectPrototype::load(
            &instance,
            &device,
            &data,
            "res\\models\\MONKEY.obj",
            lit_shader_hash,
            position,
            view,
            proj,
            (image_hash, sampler_hash),
            light_buffers.clone(),
            vec![camera.get_set_layout()],
            "monkey".to_string(),
        );
        unsafe { monkey.generate_vertex_buffer(&instance, &device, &data) };
        unsafe { monkey.generate_index_buffer(&instance, &device, &data) };

        data.objects.push(Box::new(monkey));

        let position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

        let mut plane = Quad::new(
            &instance,
            &device,
            &data,
            [
                vec3(-1.5, 0.0, -1.5),
                vec3(1.5, 0.0, -1.5),
                vec3(-1.5, 0.0, 1.5),
                vec3(1.5, 0.0, 1.5),
            ],
            vec3(0.0, 1.0, 0.0),
            lit_shader_hash,
            (image_hash, sampler_hash),
            light_buffers.clone(),
            position,
            view,
            proj,
            vec![camera.get_set_layout()],
            "Quad".to_string(),
        );
        plane.generate(&instance, &device, &data);

        data.objects.push(Box::new(plane));

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
    /// Do not call this function if the window is minimised
    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device
                .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        self.update(image_index);
        self.update_command_buffer(image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[in_flight_fence])?;

        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
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

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        begin_command_label(
            &self.instance,
            command_buffer,
            "Main Object Pass".to_string(),
            [0.0, 0.1, 0.5, 1.0],
        );

        self.device.cmd_begin_render_pass(
            command_buffer,
            &info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
        );

        let secondary_command_buffers = (0..self.data.objects.len())
            .map(|i| self.update_secondary_command_buffer(image_index, i))
            .collect::<Result<Vec<_>, _>>()?;

        self.device
            .cmd_execute_commands(command_buffer, &secondary_command_buffers);

        self.device.cmd_end_render_pass(command_buffer);

        self.device.end_command_buffer(command_buffer)?;

        end_command_label(&self.instance, command_buffer);

        Ok(())
    }

    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        self.data
            .secondary_command_buffers
            .resize_with(image_index + 1, Vec::new);
        let command_buffers = &mut self.data.secondary_command_buffers[image_index];
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.device.begin_command_buffer(command_buffer, &info)?;

        self.data.objects[model_index].draw(
            &self.instance,
            &self.device,
            command_buffer,
            image_index,
            self.data.camera.get_descriptor_sets()[image_index],
        );

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    fn update(&mut self, image_index: usize) {
        self.data.camera.calculate_view(&self.device, image_index);

        let vp = self.data.camera.proj() * self.data.camera.view();

        let mut objects = self.data.objects.clone();

        objects.iter_mut().enumerate().for_each(|(id, o)| {
            o.update(&self.device, &self.data, &self.stats, image_index, vp, id)
        });

        self.data.objects = objects;
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.data.recreated = true;

        self.device.device_wait_idle()?;
        self.destroy_swapchain();

        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;

        self.data.render_pass = generate_render_pass(
            &mut self.data.subpass_data,
            &self.data.attachments,
            &self.device,
        )?;

        let image_data = self
            .data
            .attachments
            .iter()
            .map(|a| {
                let (usage, aspect) = match a.final_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        vk::ImageAspectFlags::COLOR,
                    ),
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        vk::ImageAspectFlags::DEPTH,
                    ),
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        vk::ImageAspectFlags::DEPTH,
                    ),
                    vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL => (
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                        vk::ImageAspectFlags::STENCIL,
                    ),
                    _ => (
                        vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        vk::ImageAspectFlags::COLOR,
                    ),
                };

                (*a, usage, aspect)
            })
            .collect::<Vec<_>>();

        self.data.images =
            generate_render_pass_images(image_data, &self.data, &self.instance, &self.device);

        self.data.framebuffers = create_framebuffers(&self.data, &self.device)?;

        let mut objects = self.data.objects.clone();

        objects
            .iter_mut()
            .for_each(|o| o.recreate_swapchain(&self.device, &self.data));

        self.data.objects = objects;

        let aspect =
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32;

        self.data.camera.set_aspect(aspect);
        self.data.camera.calculate_proj(&self.device);

        create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));

        self.data
            .images
            .iter()
            .for_each(|i| i.destroy(&self.device));

        self.data
            .objects
            .iter()
            .for_each(|o| o.destroy_swapchain(&self.device));

        self.device.destroy_render_pass(self.data.render_pass, None);

        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

    /// This function will destroy all vulkan objects in the renderer
    ///
    /// # Safety
    /// This function **MUST** be called befoore the end of the program
    /// and **MUST** be the last function called on the renderer
    pub unsafe fn destroy(&mut self) {
        self.destroy_swapchain();
        self.data
            .command_pools
            .iter()
            .for_each(|p| self.device.destroy_command_pool(*p, None));

        self.device
            .destroy_command_pool(self.data.command_pool, None);

        let mut objects = self.data.objects.clone();
        objects.iter_mut().for_each(|o| o.destroy(&self.device));
        self.data.objects = objects;

        self.data
            .shaders
            .iter()
            .for_each(|(_, s)| s.destroy(&self.device));
        self.data
            .samplers
            .iter()
            .for_each(|(_, s)| self.device.destroy_sampler(*s, None));
        self.data
            .textures
            .iter()
            .for_each(|(_, t)| t.destroy(&self.device));

        self.data.camera.destroy(&self.device);

        self.data
            .point_lights
            .iter()
            .for_each(|l| l.iter().for_each(|b| b.destroy(&self.device)));

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
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Default, Clone)]
pub struct RendererData {
    // Debug
    messenger: vk::DebugUtilsMessengerEXT,

    // Surface
    surface: vk::SurfaceKHR,

    // Physical Device / Logical Device
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,

    // Pipeline
    render_pass: vk::RenderPass,
    subpass_number: usize,
    attachments: Vec<AttachmentDescription>,
    subpass_data: Vec<SubpassData>,

    // Framebuffers
    images: Vec<Image>,
    framebuffers: Vec<vk::Framebuffer>,

    // Command Buffers
    command_pool: vk::CommandPool,
    command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,

    // objects
    shaders: HashMap<u64, Box<dyn Shader>>,
    samplers: HashMap<u64, vk::Sampler>,
    textures: HashMap<u64, Image>,

    objects: Vec<Box<dyn Renderable>>,
    camera: Box<dyn Camera>,
    point_lights: Vec<Vec<BufferWrapper>>,

    // Semaphores
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,

    // Fences
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,

    // MSAA
    msaa_samples: vk::SampleCountFlags,

    recreated: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderStats {
    start: Instant,

    delta: Duration,
    delda_start: Instant,

    frame: u64,
}

unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut RendererData,
) -> Result<Instance> {
    // Application Info

    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Electrum Engine\0")
        .application_version(vk::make_version(0, 1, 0))
        .engine_name(b"Electrum Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 3, 0));

    // Layers

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Extensions

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    let flags = vk::InstanceCreateFlags::empty();

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Create

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    // Messenger

    if VALIDATION_ENABLED {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

#[derive(Debug, Error)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);

unsafe fn pick_physical_device(instance: &Instance, data: &mut RendererData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping physical device: {}: {}",
                properties.device_name, error
            );
        } else {
            info!("Selected physical device: {}.", properties.device_name);
            data.physical_device = physical_device;
            data.msaa_samples = get_max_msaa_samples(instance, data);

            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &RendererData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, data, physical_device)?;
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
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
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
        vk::SampleCountFlags::_64,
        vk::SampleCountFlags::_32,
        vk::SampleCountFlags::_16,
        vk::SampleCountFlags::_8,
        vk::SampleCountFlags::_4,
        vk::SampleCountFlags::_2,
    ]
    .iter()
    .cloned()
    .find(|c| counts.contains(*c))
    .unwrap_or(vk::SampleCountFlags::_1)
}

unsafe fn create_logical_device(instance: &Instance, data: &mut RendererData) -> Result<Device> {
    // Queue Create Infos

    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Layers

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    // Extensions

    let extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Features

    // change this for other features
    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .fill_mode_non_solid(true)
        .wide_lines(true);

    // Create

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    // Queues
    let graphics_queue = device.get_device_queue(indices.graphics, 0);
    set_object_name(
        instance,
        &device,
        "Graphics Queue".to_string(),
        vk::ObjectType::QUEUE,
        graphics_queue.as_raw() as u64,
    )?;
    data.graphics_queue = graphics_queue;

    let present_queue = device.get_device_queue(indices.present, 0);
    set_object_name(
        instance,
        &device,
        "Present Queue".to_string(),
        vk::ObjectType::QUEUE,
        graphics_queue.as_raw() as u64,
    )?;
    data.present_queue = present_queue;

    Ok(device)
}

unsafe fn get_depth_format(instance: &Instance, data: &RendererData) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        data,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
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
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for i in 0..MAX_FRAMES_IN_FLIGHT {
        let image_available_semaphor = device.create_semaphore(&semaphore_info, None)?;
        set_object_name(
            instance,
            device,
            format!("Image Available Semaphore {}", i),
            vk::ObjectType::SEMAPHORE,
            image_available_semaphor.as_raw(),
        )?;
        data.image_available_semaphores
            .push(image_available_semaphor);

        let render_finished_semaphor = device.create_semaphore(&semaphore_info, None)?;
        set_object_name(
            instance,
            device,
            format!("Render Finished Semaphore {}", i),
            vk::ObjectType::SEMAPHORE,
            render_finished_semaphor.as_raw(),
        )?;
        data.render_finished_semaphores
            .push(render_finished_semaphor);

        let in_flight_fence = device.create_fence(&fence_info, None)?;
        set_object_name(
            instance,
            device,
            format!("In Flight Fence {}", i),
            vk::ObjectType::FENCE,
            in_flight_fence.as_raw(),
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
        instance: &Instance,
        data: &RendererData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, _) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
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
        instance: &Instance,
        data: &RendererData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

fn get_hash<T>(object: &T) -> u64
where
    T: Hash,
{
    let mut hasher = DefaultHasher::new();
    object.hash(&mut hasher);
    hasher.finish()
}

fn set_object_name(
    instance: &Instance,
    device: &Device,
    name: String,
    object_type: vk::ObjectType,
    object_handle: u64,
) -> Result<()> {
    if VALIDATION_ENABLED {
        let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_name((name + "\0").as_bytes())
            .object_type(object_type)
            .object_handle(object_handle)
            .build();

        unsafe { instance.set_debug_utils_object_name_ext(device.handle(), &name_info) }?
    }

    Ok(())
}

fn begin_command_label(
    instance: &Instance,
    command_buffer: vk::CommandBuffer,
    name: String,
    colour: [f32; 4],
) {
    if VALIDATION_ENABLED {
        let info = vk::DebugUtilsLabelEXT::builder()
            .label_name((name + "\0").as_bytes())
            .color(colour)
            .build();

        unsafe { instance.cmd_begin_debug_utils_label_ext(command_buffer, &info) }
    }
}

fn end_command_label(instance: &Instance, command_buffer: vk::CommandBuffer) {
    if VALIDATION_ENABLED {
        unsafe { instance.cmd_end_debug_utils_label_ext(command_buffer) }
    }
}

/// This will place a label for debbuging applications to display at that point
/// If all the values in the `colour` array are 0.0 the colour wil be ignored
fn insert_command_label(
    instance: &Instance,
    command_buffer: vk::CommandBuffer,
    name: String,
    colour: [f32; 4],
) {
    if VALIDATION_ENABLED {
        let info = vk::DebugUtilsLabelEXT::builder()
            .label_name((name + "\0").as_bytes())
            .color(colour)
            .build();

        unsafe { instance.cmd_insert_debug_utils_label_ext(command_buffer, &info) };
    }
}
