use electrum_engine::vertices::PCTVertex;
use glam::{vec3, Mat4, Quat, Vec3};
use std::f32::consts::PI;

use electrum_engine::{
    Camera, Image, LightGroup, Material, MipLevels, ObjectPrototype, PipelineMeshSettings, PointLight, Projection, Quad, Renderer, RendererData, SimpleCamera, SubPassRenderData, VFShader, Vertex
};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

use vulkanalia::prelude::v1_2::*;

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
        let mut renderer = Renderer::create(&window).unwrap();
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
                }
            }
            WindowEvent::CloseRequested => {
                // Destroy the vulkan app
                unsafe {
                    renderer.device.device_wait_idle().unwrap();
                }
                unsafe { renderer.destroy() };
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if !self.minimized {
                    unsafe { renderer.render(&window) }.unwrap();
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn pre_load_objects(instance: &Instance, device: &Device, data: &mut RendererData) {
    let aspect = data.swapchain_extent.width as f32 / data.swapchain_extent.height as f32;

    let projection = Projection::new(PI / 4.0, aspect, 0.1, 100.0);
    let mut camera = SimpleCamera::new(
        &instance,
        &device,
        data,
        vec3(0.0, 4.0, 4.0),
        vec3(0.0, 0.0, 0.0),
        projection,
    );
    camera.look_at(vec3(0.0, 0.0, 0.0), Vec3::NEG_Y);

    let camera_id = data.cameras.push(Box::new(camera.clone()));

    let view = camera.view();
    let proj = camera.proj();

    let red_light = PointLight::new(vec3(3.0, 3.0, 0.0), vec3(1.0, 0.0, 0.0), 5.0);
    let red_light_id = data.point_light_data.push(red_light);

    let blue_light = PointLight::new(vec3(-3.0, 3.0, 0.0), vec3(0.0, 1.0, 1.0), 5.0);
    let blue_light_id = data.point_light_data.push(blue_light);

    let light_group = LightGroup::new(
        &instance,
        &device,
        data,
        vec![red_light_id, blue_light_id],
        10,
    );

    let light_group_id = data.light_groups.push(light_group.clone());

    let position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

    let lit_shader = VFShader::builder(&instance, &device, "Lit".to_string())
        .load_vertex("res\\shaders\\test_lit.vert.spv")
        .load_fragment("res\\shaders\\test_lit.frag.spv")
        .build();

    let lit_shader_id = data.shaders.push(Box::new(lit_shader));

    let image = Image::from_path(
        "res\\textures\\white.png",
        MipLevels::Maximum,
        &instance,
        &device,
        &data,
        vk::Format::R8G8B8A8_SRGB,
        true,
    );

    let image_id = data.textures.push(image);

    let image_2077 = Image::from_path(
        "res\\textures\\photomode.png",
        MipLevels::Maximum,
        &instance,
        &device,
        &data,
        vk::Format::R8G8B8A8_SRGB,
        true,
    );

    let image_2077_id = data.textures.push(image_2077);

    let mut monkey = ObjectPrototype::load(
        &instance,
        &device,
        data,
        "res\\models\\MONKEY.obj",
        position,
        view,
        proj,
        image_id,
        "monkey".to_string(),
    );
    monkey.generate_vertex_buffer(&instance, &device, &data);
    monkey.generate_index_buffer(&instance, &device, &data);

    let monkey_id = data.objects.push(Box::new(monkey));

    let position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

    let mut plane = Quad::new(
        &instance,
        &device,
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
        view,
        proj,
        "Quad".to_string(),
    );
    plane.generate(&instance, &device, &data);

    let plane_id = data.objects.push(Box::new(plane));

    let shadow_position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

    let mut shadow_plane = Quad::new(
        &instance,
        &device,
        data,
        [
            vec3(-1.5, 0.0, -1.5),
            vec3(1.5, 0.0, -1.5),
            vec3(-1.5, 0.0, 1.5),
            vec3(1.5, 0.0, 1.5),
        ],
        vec3(0.0, 1.0, 0.0),
        image_2077_id,
        shadow_position,
        view,
        proj,
        "Shadow Quad".to_string(),
    );
    shadow_plane.generate(&instance, &device, &data);

    let shadow_plane_id = data.objects.push(Box::new(shadow_plane));

    let quad_mesh_settings = PipelineMeshSettings {
        binding_descriptions: PCTVertex::binding_descriptions(),
        attribute_descriptions: PCTVertex::attribute_descriptions(),
        front_face: vk::FrontFace::CLOCKWISE,
        ..Default::default()
    };

    let bindings = vec![
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
    ];

    let shadow_material = Material::new(device, data, bindings.clone(), vec![], lit_shader_id, quad_mesh_settings.clone(), vec![camera.get_set_layout(), light_group.get_set_layout()], 0);
    let shadow_mat_id = data.materials.push(shadow_material);

    let material = Material::new(device, data, bindings.clone(), vec![], lit_shader_id, quad_mesh_settings, vec![camera.get_set_layout(), light_group.get_set_layout()], 1);
    let mat_id = data.materials.push(material);

    let render_data_0 = SubPassRenderData::new(
        0,
        vec![(shadow_plane_id, shadow_mat_id)],
        camera_id,
        light_group_id
    );

    let render_data_1 = SubPassRenderData::new(
        1,
        vec![(plane_id, mat_id), (monkey_id, mat_id)],
        camera_id,
        light_group_id,
    );
    
    data.subpass_render_data = vec![render_data_0, render_data_1];

    let mut render_data = data.subpass_render_data.clone();
    render_data
        .iter_mut()
        .for_each(|s| s.setup_command_buffers(&device, &data));
    data.subpass_render_data = render_data;
}
