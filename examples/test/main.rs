use std::f32::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};
use glam::{vec3, Mat4, Quat, Vec3};

use electrum_engine::{create_texture_sampler, Camera, Image, MipLevels, ObjectPrototype, PointLight, PointLights, Projection, Quad, Renderer, RendererData, Shader, SimpleCamera, VFShader};

use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkanalia::prelude::v1_2::*;

fn main() {
    tracing_subscriber::fmt::init();

    // Window

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Electrum Engine")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap();

    // App

    let mut renderer = Renderer::create(&window).unwrap();

    // load all objects
    pre_load_objects(&renderer.instance, &renderer.device, &mut renderer.data);


    let mut destroying = false;
    let mut minimized = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::MainEventsCleared if !destroying && !minimized => {
                unsafe { renderer.render(&window) }.unwrap()
            }

            Event::WindowEvent { event, .. } => {
                (*control_flow, minimized, destroying) =
                    window_event(&window, event, *control_flow, &mut renderer);
            }
            _ => {}
        }
    });
}

fn window_event(
    window: &Window,
    event: WindowEvent,
    mut control_flow: ControlFlow,
    app: &mut Renderer,
) -> (ControlFlow, bool, bool) {
    let mut minimized = false;
    let mut destroying = false;
    match event {
        WindowEvent::KeyboardInput {
            input:
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode,
                    ..
                },
            ..
        } => match virtual_keycode {
            Some(VirtualKeyCode::F11) => {
                if window.fullscreen().is_none() {
                    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
                        window.current_monitor(),
                    )));
                    app.resized = true;
                } else {
                    window.set_fullscreen(None);
                    app.resized = true;
                }
            }
            _ => {}
        },
        WindowEvent::Resized(size) => {
            if size.width == 0 || size.height == 0 {
                minimized = true;
            } else {
                minimized = false;
                app.resized = true;
            }
        }
        WindowEvent::CloseRequested => {
            // Destroy the vulkan app
            destroying = true;
            control_flow = ControlFlow::Exit;
            unsafe {
                app.device.device_wait_idle().unwrap();
            }
            unsafe { app.destroy() };
        }
        _ => {}
    }
    (control_flow, minimized, destroying)
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

    data.camera = Box::new(camera.clone());

    let view = data.camera.view();
    let proj = data.camera.proj();

    let red_light = PointLight::new(vec3(3.0, 3.0, 0.0), vec3(1.0, 0.0, 0.0), 5.0);
    let red_light_hash = get_hash(&red_light);

    data.point_light_data.insert(red_light_hash, red_light);

    let blue_light = PointLight::new(vec3(-3.0, 3.0, 0.0), vec3(0.0, 0.0, 1.0), 5.0);
    let blue_light_hash = get_hash(&blue_light);

    data.point_light_data.insert(blue_light_hash, blue_light);


    data.point_lights = PointLights::new(
        &instance,
        &device,
        &data,
        vec![red_light_hash, blue_light_hash],
        10,
    );

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
        vec![camera.get_set_layout(), data.point_lights.get_set_layout()],
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
        position,
        view,
        proj,
        vec![camera.get_set_layout(), data.point_lights.get_set_layout()],
        "Quad".to_string(),
    );
    plane.generate(&instance, &device, &data);

    data.objects.push(Box::new(plane));
}

fn get_hash<T>(object: &T) -> u64
where
    T: Hash,
{
    let mut hasher = DefaultHasher::new();
    object.hash(&mut hasher);
    hasher.finish()
}
