use electrum_engine::vertices::PCTVertex;
use glam::{vec3, Mat4, Quat, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::f32::consts::PI;
use std::vec;

use electrum_engine::{
    get_depth_format, Attachment, AttachmentSize, BasicMaterial, BindPoint, Camera, FrameGraph, GraphicsShader, Image, LightGroup, Mesh, MipLevels, Pass, PipelineMeshState, PointLight, Projection, Renderer, RendererData, Resource, ResourceType, ResourceUsage, Shader, SimpleCamera, SubpassPipelineState, Vertex
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
    let depth_clear = vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    };

    let color_clear = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let mut frame_graph = FrameGraph::new("Frame Graph");
    frame_graph.add_pass(
        Pass::new("Main Draw", BindPoint::Graphics)
            .create_resource(
                Resource::new(
                    "Scene Depth",
                    Some(depth_clear),
                    ResourceType::Attachment(Attachment {
                        width: AttachmentSize::Relative(1.0),
                        height: AttachmentSize::Relative(1.0),
                        format: get_depth_format(&instance, &data).unwrap(),
                        samples: data.msaa_samples,
                        ..Attachment::depth()
                    })
                ),
                ResourceUsage::Write,
            )
            .create_resource(
                Resource::new(
                    "Scene Color",
                    Some(color_clear),
                    ResourceType::Attachment(Attachment {
                        width: AttachmentSize::Relative(1.0),
                        height: AttachmentSize::Relative(1.0),
                        format: data.swapchain_format,
                        samples: data.msaa_samples,
                        ..Attachment::color()
                    })
                ),
            ResourceUsage::Write,
            )
            .create_resource(
                Resource::new(
                    "Swapchain output",
                    Some(color_clear),
                    ResourceType::Attachment(Attachment {
                        width: AttachmentSize::Relative(1.0),
                        height: AttachmentSize::Relative(1.0),
                        format: data.swapchain_format,
                        ..Attachment::present()
                    })
                ), 
                ResourceUsage::Write,
            )
            .build(),
    );

    data.render_passes = frame_graph.generate_render_pass(instance, device, data);
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
        vec![(0, vec![(0, vec![0, 1])])],
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
        vec![(0, vec![(0, vec![0, 1])])],
        "Quad".to_string(),
    );

    let plane_id = data.objects.push(plane);

    let mesh_state = PipelineMeshState::new(
        PCTVertex::binding_descriptions(),
        PCTVertex::attribute_descriptions(),
        false,
        vk::PrimitiveTopology::TRIANGLE_LIST,
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
        data.render_passes[0].render_pass,
        0
    );
    let mat_id = data.materials.push(Box::new(material));
    
    data.render_passes[0].subpasses[0].camera = camera_id;
    data.render_passes[0].subpasses[0].objects = vec![(plane_id, mat_id), (monkey_id, mat_id)];
}
