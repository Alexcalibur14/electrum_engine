use electrum_engine::vertices::PCTVertex;
use glam::{vec3, Mat4, Quat, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::f32::consts::PI;
use std::vec;

use electrum_engine::{
    get_depth_format, Attachment, AttachmentSize, AttachmentUse, BasicMaterial, Camera, GraphicsShader, Image, LightGroup, Mesh, MipLevels, PipelineMeshState, PointLight, Projection, RenderPassBuilder, Renderer, RendererData, Shader, SimpleCamera, SubpassLayoutBuilder, SubpassPipelineState, SubpassRenderData, Vertex
};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

use ash::vk;
use ash::{Device, Instance};

fn main() {
    tracing_subscriber::fmt::init();

    // Window

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App {
        window: None,
        renderer: None,
        minimized: false,
    };
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    minimized: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        let size = window.inner_size();
        let mut renderer = Renderer::create(window.display_handle().unwrap(), window.window_handle().unwrap(), size.width, size.height).unwrap();
        setup_renderpass(&renderer.instance, &renderer.device, &mut renderer.data);
        pre_load_objects(&renderer.instance, &renderer.device, &mut renderer.data);

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let window = self.window.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: key,
                        ..
                    },
                ..
            } => match key {
                Key::Named(NamedKey::F11) => {
                    if window.fullscreen().is_none() {
                        window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
                            window.current_monitor(),
                        )));
                        renderer.resized = true;
                    } else {
                        window.set_fullscreen(None);
                        renderer.resized = true;
                    }
                }
                _ => {}
            },
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    self.minimized = true;
                } else {
                    self.minimized = false;
                    renderer.resized = true;

                    
                    let aspect = size.width as f32 / size.height as f32;

                    renderer.data
                        .cameras
                        .iter_mut()
                        .for_each(|c| c.set_aspect(aspect));
                    renderer.data
                        .cameras
                        .iter_mut()
                        .for_each(|c| c.calculate_proj(&renderer.device));
                }
            }
            WindowEvent::CloseRequested => {
                // Destroy the vulkan app
                unsafe {
                    renderer.device.device_wait_idle().unwrap();
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if !self.minimized {
                    let size = window.inner_size();
                    unsafe { renderer.render(size.width, size.height) }.unwrap();
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn setup_renderpass(instance: &Instance, device: &Device, data: &mut RendererData) {
    let mut attachments = vec![
        Attachment {
            format: get_depth_format(&instance, &data).unwrap(),
            x: AttachmentSize::Absolute(1024),
            y: AttachmentSize::Absolute(1024),
            ..Attachment::template_depth()
        },
        Attachment {
            format: data.swapchain_format,
            sample_count: data.msaa_samples,
            ..Attachment::template_colour()
        },
        Attachment {
            format: get_depth_format(&instance, &data).unwrap(),
            sample_count: data.msaa_samples,
            ..Attachment::template_depth()
        },
        Attachment {
            format: data.swapchain_format,
            ..Attachment::template_present()
        },
    ];

    attachments.iter_mut().for_each(|a| a.generate());

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

    let subpass_1 = SubpassLayoutBuilder::new(vk::PipelineBindPoint::GRAPHICS)
        .add_depth_stencil_attachment(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();

    let subpass_2 = SubpassLayoutBuilder::new(vk::PipelineBindPoint::GRAPHICS)
        .add_color_attachment(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .add_depth_stencil_attachment(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .add_resolve_attachment(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let mut render_pass_builder0 = RenderPassBuilder::new(false, String::from("Shadow"))
        .add_attachment(attachments[0], AttachmentUse::Depth, depth_clear_value)
        .add_subpass(&subpass_1, &[(0, None)])
        .build();
    
    let mut render_pass_builder1 = RenderPassBuilder::new(true, String::from("Lit"))
        .add_attachment(attachments[1], AttachmentUse::Color, color_clear_value)
        .add_attachment(attachments[2], AttachmentUse::Depth, depth_clear_value)
        .add_attachment(attachments[3], AttachmentUse::Color, color_clear_value)
        .add_subpass(&subpass_2, &[(0, None), (1, None), (2, None)])
        .build();

    let (render_pass, framebuffers) = render_pass_builder0.create_render_pass(instance, device, data).unwrap();
    data.render_passes.push(render_pass);
    data.framebuffers.push(framebuffers);
    data.render_pass_builders.push(render_pass_builder0);

    let (render_pass, framebuffers) = render_pass_builder1.create_render_pass(instance, device, data).unwrap();
    data.render_passes.push(render_pass);
    data.framebuffers.push(framebuffers);
    data.render_pass_builders.push(render_pass_builder1);
}

fn pre_load_objects(instance: &Instance, device: &Device, data: &mut RendererData) {
    let aspect = data.swapchain_extent.width as f32 / data.swapchain_extent.height as f32;

    let projection = Projection::new(PI / 4.0, aspect, 0.1, 100.0);
    let mut camera = SimpleCamera::new(
        instance,
        device,
        data,
        vec3(0.0, 4.0, 4.0),
        vec3(0.0, 0.0, 0.0),
        projection,
    );
    camera.look_at(vec3(0.0, 0.0, 0.0), Vec3::NEG_Y);

    let camera_id = data.cameras.push(Box::new(camera.clone()));

    let red_light = PointLight::new(vec3(3.0, 3.0, 0.0), vec3(1.0, 0.0, 0.0), 5.0);
    let red_light_id = data.point_light_data.push(red_light);

    let blue_light = PointLight::new(vec3(-3.0, 3.0, 0.0), vec3(0.0, 1.0, 1.0), 5.0);
    let blue_light_id = data.point_light_data.push(blue_light);

    let light_group = LightGroup::new(
        instance,
        device,
        data,
        vec![red_light_id, blue_light_id],
        5,
    );

    let _ = data.other_descriptors.push(Box::new(light_group.clone()));

    let position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

    let lit_shader = GraphicsShader::builder(instance, device, "Lit".to_string())
        .load_vertex("res\\shaders\\test_lit.vert.spv")
        .load_fragment("res\\shaders\\test_lit.frag.spv")
        .build();

    let lit_state = lit_shader.state();

    data.shaders.push(Box::new(lit_shader));

    let shadow_shader = GraphicsShader::builder(instance, device, "Shadow".to_string())
        .load_vertex("res\\shaders\\test_shadow.vert.spv") 
        .build();

    let shadow_state = shadow_shader.state();

    data.shaders.push(Box::new(shadow_shader));

    let image = Image::from_path(
        instance,
        device,
        data,
        "res\\textures\\white.png",
        MipLevels::Maximum,
        vk::Format::R8G8B8A8_SRGB,
        true,
    );

    let image_id = data.textures.push(image);

    let image_2077 = Image::from_path(
        instance,
        device,
        data,
        "res\\textures\\photomode.png",
        MipLevels::Maximum,
        vk::Format::R8G8B8A8_SRGB,
        true,
    );

    let image_2077_id = data.textures.push(image_2077);

    let monkey = Mesh::from_obj(
        instance,
        device,
        data,
        "res\\models\\MONKEY.obj",
        position,
        image_id,
        vec![(0, vec![(0, vec![0])]), (1, vec![(0, vec![0, 1])])],
        "monkey".to_string(),
    );

    let monkey_id = data.objects.push(monkey);

    let position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

    let plane = Mesh::new_quad(
        instance,
        device,
        data,
        [
            vec3(-1.5, 0.0, -1.5),
            vec3(1.5, 0.0, -1.5),
            vec3(-1.5, 0.0, 1.5),
            vec3(1.5, 0.0, 1.5),
        ],
        vec3(0.0, 1.0, 0.0),
        image_2077_id,
        position,
        vec![(0, vec![(0, vec![0])]), (1, vec![(0, vec![0, 1])])],
        "Quad".to_string(),
    );

    let plane_id = data.objects.push(plane);

    let mesh_state = PipelineMeshState::new(
        PCTVertex::binding_descriptions(),
        PCTVertex::attribute_descriptions(),
        false,
        vk::PrimitiveTopology::TRIANGLE_LIST,
    );

    let shadow_subpass_state = SubpassPipelineState::new(
        vec![
            vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(data.swapchain_extent.width as f32)
                .height(data.swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0),
        ],
        vec![
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: data.swapchain_extent.width, height: data.swapchain_extent.height },
            }
        ],
        vk::SampleCountFlags::TYPE_1,
        true,
        true,
        vec![],
        false,
        vk::LogicOp::COPY,
        [0.0, 0.0, 0.0, 0.0],
    );

    let subpass_state = SubpassPipelineState::new(
        vec![
            vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(data.swapchain_extent.width as f32)
                .height(data.swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0),
        ],
        vec![
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: data.swapchain_extent.width, height: data.swapchain_extent.height },
            }
        ],
        data.msaa_samples,
        true,
        true,
        vec![
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)
        ],
        false,
        vk::LogicOp::COPY,
        [0.0, 0.0, 0.0, 0.0],
    );

    let lit_bindings = vec![
        vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
        ],
        vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ],
    ];

    let shadow_bindings = vec![
        vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
        ],
    ];

    let shadow_material = BasicMaterial::new(
        instance,
        device,
        data,
        shadow_bindings,
        vec![],
        shadow_state,
        shadow_subpass_state,
        mesh_state.clone(),
        vec![camera.get_set_layout()],
        vec![0],
        data.render_passes[0],
        0
    );
    let shadow_mat_id = data.materials.push(Box::new(shadow_material));

    let material = BasicMaterial::new(
        instance,
        device,
        data,
        lit_bindings,
        vec![],
        lit_state,
        subpass_state,
        mesh_state,
        vec![camera.get_set_layout(), light_group.get_set_layout()],
        vec![0, 1],
        data.render_passes[1],
        0
    );
    let mat_id = data.materials.push(Box::new(material));

    let render_data_0 = SubpassRenderData::new(
        0,
        0,
        vec![(plane_id, shadow_mat_id), (monkey_id, shadow_mat_id)],
        camera_id,
        "Shadow Subpass".to_string(),
    );

    let render_data_1 = SubpassRenderData::new(
        1,
        0,
        vec![(plane_id, mat_id), (monkey_id, mat_id)],
        camera_id,
        "Lighting Subpass".to_string(),
    );
    
    data.subpass_render_data = vec![vec![render_data_0], vec![render_data_1]];

    let mut render_data = data.subpass_render_data.clone();
    render_data
        .iter_mut()
        .for_each(|s| s.iter_mut().for_each(|s| s.setup_command_buffers(&device, &data)));
    data.subpass_render_data = render_data;
}
