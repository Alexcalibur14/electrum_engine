use glam::{vec3, Mat4, Quat, Vec3};
use std::f32::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};

use electrum_engine::{
    create_texture_sampler, Camera, Image, LightGroup, MipLevels, ObjectPrototype, PointLight,
    Projection, Quad, Renderer, RendererData, Shader, ShadowQuad, SimpleCamera, SubPassRenderData,
    VFShader,
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
        &data,
        vec3(0.0, 4.0, 4.0),
        vec3(0.0, 0.0, 0.0),
        projection,
    );
    camera.look_at(vec3(0.0, 0.0, 0.0), Vec3::NEG_Y);

    let camera_hash = get_hash(&camera);

    data.cameras.insert(camera_hash, Box::new(camera.clone()));

    let view = camera.view();
    let proj = camera.proj();

    let red_light = PointLight::new(vec3(3.0, 3.0, 0.0), vec3(1.0, 0.0, 0.0), 5.0);
    let red_light_hash = get_hash(&red_light);

    data.point_light_data.insert(red_light_hash, red_light);

    let blue_light = PointLight::new(vec3(-3.0, 3.0, 0.0), vec3(0.0, 1.0, 1.0), 5.0);
    let blue_light_hash = get_hash(&blue_light);

    data.point_light_data.insert(blue_light_hash, blue_light);

    let light_group = LightGroup::new(
        &instance,
        &device,
        &data,
        vec![red_light_hash, blue_light_hash],
        10,
    );

    let light_group_hash = get_hash(&light_group);

    data.light_groups
        .insert(light_group_hash, light_group.clone());

    let position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

    let lit_shader = VFShader::builder(&instance, &device, "Lit".to_string())
        .compile_vertex("res\\shaders\\test_lit.vert.glsl")
        .compile_fragment("res\\shaders\\test_lit.frag.glsl")
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

    let image_2077 = Image::from_path(
        "res\\textures\\photomode.png",
        MipLevels::Maximum,
        &instance,
        &device,
        &data,
        vk::Format::R8G8B8A8_SRGB,
    );

    let sampler_2077 = unsafe {
        create_texture_sampler(
            &instance,
            &device,
            &image_2077.mip_level,
            "2077 sampler".to_string(),
        )
    }
    .unwrap();

    let sampler_2077_hash = get_hash(&sampler_2077);
    data.samplers.insert(sampler_2077_hash, sampler_2077);

    let image_2077_hash = get_hash(&image_2077);
    data.textures.insert(image_2077_hash, image_2077);

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
        vec![camera.get_set_layout(), light_group.get_set_layout()],
        "monkey".to_string(),
    );
    unsafe { monkey.generate_vertex_buffer(&instance, &device, &data) };
    unsafe { monkey.generate_index_buffer(&instance, &device, &data) };

    let monkey_hash = get_hash(&monkey);
    data.objects.insert(monkey_hash, Box::new(monkey));

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
        (image_2077_hash, sampler_2077_hash),
        position,
        view,
        proj,
        vec![camera.get_set_layout(), light_group.get_set_layout()],
        "Quad".to_string(),
    );
    plane.generate(&instance, &device, &data);

    let plane_hash = get_hash(&plane);
    data.objects.insert(plane_hash, Box::new(plane));

    let shadow_position = Mat4::from_rotation_translation(Quat::IDENTITY, vec3(0.0, 0.0, 0.0));

    let mut shadow_plane = ShadowQuad::new(
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
        (image_2077_hash, sampler_2077_hash),
        shadow_position,
        view,
        proj,
        vec![camera.get_set_layout(), light_group.get_set_layout()],
        "Shadow Quad".to_string(),
    );
    shadow_plane.generate(&instance, &device, &data);

    let shadow_plane_hash = get_hash(&shadow_plane);
    data.objects
        .insert(shadow_plane_hash, Box::new(shadow_plane));

    let render_data_0 =
        SubPassRenderData::new(0, vec![shadow_plane_hash], camera_hash, light_group_hash);
    let render_data_1 = SubPassRenderData::new(
        1,
        vec![plane_hash, monkey_hash],
        camera_hash,
        light_group_hash,
    );
    data.subpass_render_data = vec![render_data_0, render_data_1];

    let mut render_data = data.subpass_render_data.clone();
    render_data
        .iter_mut()
        .for_each(|s| s.setup_command_buffers(&device, &data));
    data.subpass_render_data = render_data;
}

fn get_hash<T>(object: &T) -> u64
where
    T: Hash,
{
    let mut hasher = DefaultHasher::new();
    object.hash(&mut hasher);
    hasher.finish()
}
