use std::collections::VecDeque;
use std::f32::consts::PI;
use std::ops::RangeInclusive;
use std::path::Path;

use ash::Device;
use ash::Instance;
use ash::vk;

use anyhow::Result;
use electrum_engine::begin_command_label;
use electrum_engine::buffer::Buffer;
use electrum_engine::buffer::BufferType;
use electrum_engine::descriptor::DescriptorBuilder;
use electrum_engine::descriptor::update_image_binding;
use electrum_engine::end_command_label;
use electrum_engine::extra::CameraData;
use electrum_engine::extra::Light;
use electrum_engine::extra::LightData;
use electrum_engine::extra::LightType;
use electrum_engine::extra::ModelData;
use electrum_engine::model::primitives::Plane;
use electrum_engine::extra::Projection;
use electrum_engine::extra::SimpleCamera;
use electrum_engine::extra::orthographic_symetric;
use electrum_engine::extra::radians;
use electrum_engine::image::AddressMode;
use electrum_engine::image::Filter;
use electrum_engine::image::Image;
use electrum_engine::image::MipLevels;
use electrum_engine::image::copy_image_to_image;
use electrum_engine::model::OBJVertex;
use electrum_engine::model::Object;
use electrum_engine::model::basic_obj_loader;
use electrum_engine::model::primitives::UVSphere;
use electrum_engine::shader::DepthStencilData;
use electrum_engine::shader::MultisampleData;
use electrum_engine::shader::RasterizationData;
use electrum_engine::shader::SlangShader;
use electrum_engine::shader::create_basic_slang_graphics_pipeline;
use electrum_engine::task_graph::*;
use electrum_engine::{RenderStats, Renderer, RendererData};
use electrum_engine::Vertex;
use electrum_engine_macros::Vertex;
use glam::Mat4;
use glam::Quat;
use glam::Vec3;
use glam::vec3;
use glam::vec4;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
use winit::event::KeyEvent;
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::{ControlFlow, EventLoopBuilder}, platform::x11::EventLoopBuilderExtX11, window::Window};

fn main() -> Result<()> {
    let layer = tracing_subscriber::fmt::layer().with_filter(LevelFilter::INFO);
    tracing_subscriber::registry().with(layer).init();

    info!("Starting..");
    let event_loop = EventLoopBuilder::default()
        .with_x11() // egui selection is offset from cursor on wayland and renderdoc does not work
        .build().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    
    Ok(())
}

#[derive(Default)]
struct App<'a> {
    window: Option<Window>,
    renderer: Option<Renderer<'a>>,

    egui_data: EguiData,

    camera: SimpleCamera,
    light: Light,
    colour_correction: ColourCorrection,
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(Window::default_attributes()
            .with_title("Electrum Renderer Example")
            .with_resizable(true)
        ).unwrap();

        let mut renderer = Renderer::new(&window, true).unwrap();

        setup_render_graph(&renderer.instance, &renderer.device, &mut renderer.data);

        let white_texture = Image::load_from_file(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            "examples/test/res/textures/white.png",
            MipLevels::One,
            Some((Filter::MIN_LINEAR, AddressMode::REPEAT))
        );

        let uv_texture = Image::load_from_file(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            "examples/test/res/textures/UV_test.png",
            MipLevels::One,
            Some((Filter::MIN_LINEAR, AddressMode::REPEAT))
        );

        let mut main_shader = SlangShader::new("lit", Path::new("examples/test/res/shaders/lit.slang"));
        main_shader.load_and_compile(&renderer.device, &mut renderer.data);

        let projection = Projection::new(radians(45.0), renderer.width as f32 / renderer.height as f32, 0.01, 100.0);

        let view = Mat4::look_at_rh(vec3(0.0, 3.0, 5.0), vec3(0.0, 0.0, 0.0), Vec3::Y);
        let camera = SimpleCamera::new_view(&renderer.instance, &renderer.device, &mut renderer.data, view, projection);

        self.camera = camera;

        let model_matrix = glam::Mat4::from_scale_rotation_translation(vec3(1.0, 1.0, 1.0), Quat::IDENTITY, vec3(0.0, 0.5, 0.0));
        let model_data = ModelData::new(model_matrix);
        let object_buffer = Buffer::create_and_load(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::DeviceLocal,
            &[model_data],
            "object_matrix",
        ).unwrap();

        let (object_descriptor, _) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[object_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let monkey_material = MaterialData {
            specular: 0.5,
        };

        let material_buffer = Buffer::create_and_load(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::DeviceLocal,
            &[monkey_material],
            "plane_material"
        ).unwrap();

        let (material_descriptor, _) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[material_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::FRAGMENT)
            .bind_image(1, 1, &[white_texture.descriptor_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)], vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let mut monkey = Object::new("monkey");
        monkey.add_buffer(object_buffer, "mvp");
        monkey.add_descriptor_set(object_descriptor, "mvp");
        monkey.add_buffer(material_buffer, "material");
        monkey.add_image(white_texture, "albedo");
        monkey.add_descriptor_set(material_descriptor, "material");

        *monkey.mesh_data_mut() = basic_obj_loader(&renderer.instance, &renderer.device, &renderer.data, "examples/test/res/models/MONKEY.obj")[0].clone();

        renderer.data.objects.push(monkey, &["main", "shadow"]);


        let model_matrix = glam::Mat4::from_scale_rotation_translation(vec3(1.0, 1.0, 1.0), Quat::IDENTITY, vec3(0.0, 0.0, 0.0));
        let model_data = ModelData::new(model_matrix);
        let object_buffer = Buffer::create_and_load(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::DeviceLocal,
            &[model_data],
            "object_matrix"
        ).unwrap();

        let (object_descriptor, _) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[object_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let plane_material = MaterialData {
            specular: 0.3,
        };

        let material_buffer = Buffer::create_and_load(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::DeviceLocal,
            &[plane_material],
            "plane_material"
        ).unwrap();

        let (material_descriptor, _) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[material_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::FRAGMENT)
            .bind_image(1, 1, &[uv_texture.descriptor_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)], vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let plane = Plane::new(&renderer.instance, &renderer.device, &mut renderer.data, 10.0, 10.0, 10, 10, &["main", "shadow"]);
        let plane_object = renderer.data.objects.get_mut(&plane.object()).unwrap();
        plane_object.add_buffer(object_buffer, "mvp");
        plane_object.add_descriptor_set(object_descriptor, "mvp");
        plane_object.add_buffer(material_buffer, "material");
        plane_object.add_image(uv_texture, "albedo");
        plane_object.add_descriptor_set(material_descriptor, "material");


        let model_matrix = glam::Mat4::from_scale_rotation_translation(vec3(1.0, 1.0, 1.0), Quat::IDENTITY, vec3(-2.0, 2.0, 0.0));
        let model_data = ModelData::new(model_matrix);
        let object_buffer = Buffer::create_and_load(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::DeviceLocal,
            &[model_data],
            "object_matrix"
        ).unwrap();

        let (object_descriptor, _) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[object_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let plane_material = MaterialData {
            specular: 0.3,
        };

        let material_buffer = Buffer::create_and_load(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::DeviceLocal,
            &[plane_material],
            "uv_sphere_material",
        ).unwrap();

        let (material_descriptor, _) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[material_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::FRAGMENT)
            .bind_image(1, 1, &[uv_texture.descriptor_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)], vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let uv_sphere = UVSphere::new(&renderer.instance, &renderer.device, &mut renderer.data, 1.0, 6, 12, &["main", "shadow"]);
        let uv_sphere_object = renderer.data.objects.get_mut(&uv_sphere.object()).unwrap();
        uv_sphere_object.add_buffer(object_buffer, "mvp");
        uv_sphere_object.add_descriptor_set(object_descriptor, "mvp");
        uv_sphere_object.add_buffer(material_buffer, "material");
        uv_sphere_object.add_descriptor_set(material_descriptor, "material");


        let light_position = vec3(3.0, 3.0, 0.0);
        let light_target = Vec3::ZERO;
        let light_direction = (light_position - light_target).normalize();
        let light_data = LightData {
            position: light_position,
            direction: light_direction,
            colour: vec3(1.0, 1.0, 1.0),
            strength: 25.0,
            light_type: LightType::Directional,
        };
        self.light = Light::new(&renderer.instance, &renderer.device, &mut renderer.data, light_data);

        let (depth_image, _) = renderer.data.task_graph.images().iter().find(|(image_data, _)| image_data.name() == "shadow_map").unwrap();

        let (shadow_map_descriptor, _) = DescriptorBuilder::new()
            .bind_image(0, 1, &[depth_image.image().descriptor_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)], vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let light_object = renderer.data.objects.get_mut(self.light.object()).unwrap();
        light_object.add_descriptor_set(shadow_map_descriptor, "shadow_map");

        let light_camera_matrix = Mat4::look_at_rh(light_position, light_target, Vec3::Y);
        let light_camera_data = CameraData {
            view: light_camera_matrix,
            proj: orthographic_symetric(10.0, 10.0, 0.01, 10.0),
            position: light_position,
        };
        let light_camera_buffer = Buffer::create_and_load(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::DeviceLocal,
            &[light_camera_data],
            "light_camera_data",
        ).unwrap();
        let (light_camera_descriptor, _) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[light_camera_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let light_object = renderer.data.objects.get_mut(self.light.object()).unwrap();
        light_object.add_buffer(light_camera_buffer, "camera_data");
        light_object.add_descriptor_set(light_camera_descriptor, "camera_data");


        let pipeline = create_basic_slang_graphics_pipeline::<OBJVertex>(
            &renderer.instance,
            &renderer.device,
            &main_shader,
            RasterizationData {
                cull_mode: vk::CullModeFlags::BACK,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                polygon_mode: vk::PolygonMode::FILL,
                line_width: 0.0,
            },
            MultisampleData {
                samples: vk::SampleCountFlags::TYPE_1,
                sample_shading_enable: false,
            },
            DepthStencilData {
                depth_test_enable: true,
                depth_write_enable: true,
                stencil_test_enable: false,
                compare_op: vk::CompareOp::GREATER,
            },
            0,
            &[vk::PipelineColorBlendAttachmentState::default().blend_enable(false).color_write_mask(vk::ColorComponentFlags::RGBA)],
            &[vk::Format::R16G16B16A16_SFLOAT],
            vk::Format::D32_SFLOAT,
            vk::Format::UNDEFINED,
            &[],
            vk::PrimitiveTopology::TRIANGLE_LIST
        );

        renderer.data.slang_shaders.push(main_shader, &["main"]);
        renderer.data.pipelines.push(pipeline, &["main"]);

        let mut shadow_shader = SlangShader::new("shadow", Path::new("examples/test/res/shaders/shadow.slang"));
        shadow_shader.load_and_compile(&renderer.device, &mut renderer.data);

        let (shadow_pipeline, layout) = create_basic_slang_graphics_pipeline::<OBJVertex>(
            &renderer.instance,
            &renderer.device,
            &shadow_shader,
            RasterizationData {
                cull_mode: vk::CullModeFlags::FRONT,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                polygon_mode: vk::PolygonMode::FILL,
                line_width: 0.0,
            },
            MultisampleData {
                samples: vk::SampleCountFlags::TYPE_1,
                sample_shading_enable: false,
            },
            DepthStencilData {
                depth_test_enable: true,
                depth_write_enable: true,
                stencil_test_enable: false,
                compare_op: vk::CompareOp::GREATER,
            },
            0,
            &[],
            &[],
            vk::Format::D32_SFLOAT,
            vk::Format::UNDEFINED,
            &[],
            vk::PrimitiveTopology::TRIANGLE_LIST,
        );

        renderer.data.slang_shaders.push(shadow_shader, &["shadow"]);
        renderer.data.pipelines.push((shadow_pipeline, layout), &["shadow"]);

        let mut debug_shader = SlangShader::new("debug", Path::new("examples/test/res/shaders/debug.slang"));
        debug_shader.load_and_compile(&renderer.device, &mut renderer.data);

        let (debug_pipeline, layout) = create_basic_slang_graphics_pipeline::<OBJVertex>(
            &renderer.instance,
            &renderer.device,
            &debug_shader,
            RasterizationData {
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                polygon_mode: vk::PolygonMode::LINE,
                line_width: 1.0,
            },
            MultisampleData {
                samples: vk::SampleCountFlags::TYPE_1,
                sample_shading_enable: false,
            },
            DepthStencilData {
                depth_test_enable: false,
                depth_write_enable: false,
                stencil_test_enable: false,
                compare_op: vk::CompareOp::GREATER,
            },
            0,
            &[vk::PipelineColorBlendAttachmentState::default().blend_enable(false).color_write_mask(vk::ColorComponentFlags::RGBA)],
            &[vk::Format::R16G16B16A16_SFLOAT],
            vk::Format::UNDEFINED,
            vk::Format::UNDEFINED,
            &[],
            vk::PrimitiveTopology::LINE_LIST,
        );

        renderer.data.slang_shaders.push(debug_shader, &["debug"]);
        renderer.data.pipelines.push((debug_pipeline, layout), &["debug"]);

        let mut cc_shader = SlangShader::new("color_correction", Path::new("examples/test/res/shaders/color_correction.slang"));
        cc_shader.load_and_compile(&renderer.device, &mut renderer.data);

        let levels = Levels {
            in_black: 0.0,
            in_white: 1.0,
            out_black: 0.0,
            out_white: 1.0,
            gamma: 1.0,
        };

        let colour_correction_data = ColourCorrection {
            levels,
            hue_shift: 0.0,
        };
        self.colour_correction = colour_correction_data;
        let cc_buffer = Buffer::create_and_load(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::HostLocal,
            &[colour_correction_data],
            "colour_correction_data"
        ).unwrap();
        let (cc_descriptor, _) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[cc_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::FRAGMENT)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let mut cc_object = Object::new("colour_correction_data");
        cc_object.add_buffer(cc_buffer, "colour_correction_data");
        cc_object.add_descriptor_set(cc_descriptor, "colour_correction_data");

        let (main_image, _) = renderer.data.task_graph.images().iter().find(|(image_data, _)| image_data.name() == "color_attachment").unwrap();

        let (cc_image_descriptor, _) = DescriptorBuilder::new()
            .bind_image(
                0,
                1,
                &[
                    main_image.image().descriptor_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                ],
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT
            )
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        cc_object.add_descriptor_set(cc_image_descriptor, "colour_correction_images");

        let (cc_pipeline, layout) = create_basic_slang_graphics_pipeline::<NullVertex>(
            &renderer.instance,
            &renderer.device,
            &cc_shader,
            RasterizationData {
                cull_mode: vk::CullModeFlags::BACK,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                polygon_mode: vk::PolygonMode::FILL,
                line_width: 1.0,
            },
            MultisampleData {
                samples: vk::SampleCountFlags::TYPE_1,
                sample_shading_enable: false,
            },
            DepthStencilData {
                depth_test_enable: false,
                depth_write_enable: false,
                stencil_test_enable: false,
                compare_op: vk::CompareOp::GREATER,
            },
            0,
            &[vk::PipelineColorBlendAttachmentState::default().blend_enable(false).color_write_mask(vk::ColorComponentFlags::RGBA)],
            &[vk::Format::R8G8B8A8_SRGB],
            vk::Format::UNDEFINED,
            vk::Format::UNDEFINED,
            &[],
            vk::PrimitiveTopology::TRIANGLE_LIST,
        );

        renderer.data.objects.push(cc_object, &["colour_correction"]);
        renderer.data.slang_shaders.push(cc_shader, &["colour_correction"]);
        renderer.data.pipelines.push((cc_pipeline, layout), &["colour_correction"]);

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let renderer = self.renderer.as_mut().unwrap();

        let stats = renderer.stats;

        let window = self.window.as_ref().unwrap();
        let _ = renderer.data.egui_winit.as_mut().unwrap().on_window_event(window, &event);

        self.egui_data.update(&stats);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                if event_loop.exiting() {
                    return;
                }

                let cc_object = renderer.data.objects.get_with_tag("colour_correction")[0];
                let cc_buffer = cc_object.get_buffer("colour_correction_data");
                cc_buffer.copy_data_into_buffer(&renderer.device, &self.colour_correction);

                unsafe { renderer.render(window, &mut |ctx| {
                    ctx.style_mut(|style| style.visuals.window_shadow = egui::epaint::Shadow::NONE);
                    let _ = egui::Window::new("Render Info")
                        .anchor(egui::Align2::LEFT_TOP, (0.0, 0.0))
                        .title_bar(false)
                        .resizable([false, false])
                        .show(ctx, |ui| {
                            ui.label("Stats");
                            ui.label(format!("Framerate: {:.2} fps", self.egui_data.get_framerate()));
                            ui.label(format!("Frame Time: {:.2} us", self.egui_data.get_frame_time()));
                            ui.label(format!("Frame number: {}", stats.frame));
                            ui.label(format!("Draw Calls: {}", stats.draw_calls));
                            ui.label(format!("Command Buffer time: {:.2} us", self.egui_data.get_cmd_buf_record_time()))
                    });

                    let _ = egui::Window::new("Colour Correction")
                        .anchor(egui::Align2::RIGHT_TOP, (0.0, 0.0))
                        .title_bar(false)
                        .resizable([false, false])
                        .show(ctx, |ui| {
                            ui.collapsing("Levels", |ui| {
                                egui::Grid::new("level_grid").show(ui, |ui| {
                                    ui.label("Black In");
                                    ui.add(egui::DragValue::new(&mut self.colour_correction.levels.in_black).speed(0.001).range(RangeInclusive::new(-1.0, 2.0)));
                                    ui.end_row();

                                    ui.label("White In");
                                    ui.add(egui::DragValue::new(&mut self.colour_correction.levels.in_white).speed(0.01).range(RangeInclusive::new(-1.0, 5.0)));
                                    ui.end_row();

                                    ui.label("Black Out");
                                    ui.add(egui::DragValue::new(&mut self.colour_correction.levels.out_black).speed(0.001).range(RangeInclusive::new(-5.0, 5.0)));
                                    ui.end_row();

                                    ui.label("White Out");
                                    ui.add(egui::DragValue::new(&mut self.colour_correction.levels.out_white).speed(0.01).range(RangeInclusive::new(-5.0, 5.0)));
                                    ui.end_row();

                                    ui.label("Gamma");
                                    ui.add(egui::DragValue::new(&mut self.colour_correction.levels.gamma).speed(0.01).range(RangeInclusive::new(-1.0, 2.0)));
                                    ui.end_row();
                                });
                            });
                            ui.label("Hue Shift");
                            ui.add(egui::Slider::new(&mut self.colour_correction.hue_shift, RangeInclusive::new(0.0, 2.0 * PI)));
                            ui.end_row();
                        });
                    })
                }.unwrap();

                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(new_size) => {
                info!("Resize requested: {}x{}, frame: {}", new_size.width, new_size.height, renderer.stats.frame);
                renderer.resized = true;
                renderer.width = new_size.width;
                renderer.height = new_size.height;

                self.camera.projection.aspect_ratio = new_size.width as f32 / new_size.height as f32;
                self.camera.projection.recalculate();
                self.camera.rebuild(&renderer.device, &renderer.data);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state, .. }, .. } => {
                match physical_key {
                    winit::keyboard::PhysicalKey::Code(key_code) => {
                        match key_code {
                            winit::keyboard::KeyCode::F11 => {
                                if state.is_pressed() {
                                    unsafe { self.renderer.as_ref().unwrap().device.device_wait_idle() }.unwrap();
    
                                    let window = self.window.as_ref().unwrap();
    
                                    match window.fullscreen() {
                                        Some(_) => {
                                            window.set_fullscreen(None);
                                            self.renderer.as_mut().unwrap().resized = true;
                                        },
                                        None => {
                                            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(window.current_monitor())));
                                            self.renderer.as_mut().unwrap().resized = true;
                                        },
                                    }
                                }
                            }
                            _ => {}
                        }
                    },
                    winit::keyboard::PhysicalKey::Unidentified(_) => {},
                }
            }
            _ => (),
        }
    }
}

fn setup_render_graph(instance: &Instance, device: &Device, data: &mut RendererData) {
    let mut task_graph = TaskGraph::new();

        let image = ImageData::new(
            ImageSize::SwapchainRelative { multiplier: 1.0 },
            vk::Format::R16G16B16A16_SFLOAT,
            MipLevels::One,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageViewType::TYPE_2D,
            1,
            vk::ImageAspectFlags::COLOR,
            Some((Filter::LINEAR, AddressMode::CLAMP_TO_EDGE)),
            "color_attachment",
        );

        let depth = ImageData::new(
            ImageSize::Swapchain,
            vk::Format::D32_SFLOAT,
            MipLevels::One,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageViewType::TYPE_2D,
            1,
            vk::ImageAspectFlags::DEPTH,
            None,
            "depth_attachment",
        );

        let shadow_map = ImageData::new(
            ImageSize::Fixed { width: 1024, height: 1024 },
            vk::Format::D32_SFLOAT,
            MipLevels::One,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageViewType::TYPE_2D,
            1,
            vk::ImageAspectFlags::DEPTH,
            Some((Filter::LINEAR, AddressMode::CLAMP_TO_EDGE)),
            "shadow_map",
        );

        let cc_image = ImageData::new(
            ImageSize::Swapchain,
            vk::Format::R8G8B8A8_SRGB,
            MipLevels::One,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageViewType::TYPE_2D,
            1,
            vk::ImageAspectFlags::COLOR,
            None,
            "color_correction",
        );

        task_graph.add_image(image);
        task_graph.add_image(depth);
        task_graph.add_image(shadow_map);
        task_graph.add_image(cc_image);

        let mut shadow_pass_node = Node::new();
        shadow_pass_node.set_depth("shadow_map", AccessType::depth_stencil_attachment_read() | AccessType::depth_stencil_attachment_write() );
        shadow_pass_node.set_task(Box::new(ShadowPass));
        task_graph.add_node(shadow_pass_node);

        let mut main_node = Node::new();
        main_node.add_attachment("color_attachment", AccessType::color_attachment_write());
        main_node.set_depth("depth_attachment", AccessType::depth_stencil_attachment_read() | AccessType::depth_stencil_attachment_write());
        main_node.add_internal_image("shadow_map", vk::ImageAspectFlags::DEPTH, AccessType::fragment_shader_sampled_read());
        main_node.set_task(Box::new(MainDraw));
        task_graph.add_node(main_node);

        let mut debug_node = Node::new();
        debug_node.add_attachment("color_attachment", AccessType::color_attachment_write());
        debug_node.set_task(Box::new(DebugDraw));
        task_graph.add_node(debug_node);

        let mut cc_node = Node::new();
        cc_node.add_internal_image("color_attachment", vk::ImageAspectFlags::COLOR, AccessType::fragment_shader_sampled_read());
        cc_node.add_attachment("color_correction", AccessType::color_attachment_write());
        cc_node.set_task(Box::new(ColourCorrectionPass));
        task_graph.add_node(cc_node);

        let mut egui_node = Node::new();
        egui_node.add_attachment("color_correction", AccessType::color_attachment_write());
        egui_node.set_task(Box::new(EguiDraw));
        task_graph.add_node(egui_node);

        let mut present_node = Node::new();
        present_node.add_attachment("color_correction", AccessType::blit_src());
        present_node.set_task(Box::new(Present));
        task_graph.add_node(present_node);

        task_graph.create_images(instance, device, data);
        data.task_graph = task_graph;
}

#[derive(Debug, Default)]
struct EguiData {
    frame_times: VecDeque<u128>,
    cmd_buf_times: VecDeque<u128>,
}

impl EguiData {
    fn update(&mut self, stats: &RenderStats) {
        if self.frame_times.len() < 20 {
            self.frame_times.push_back(stats.delta.as_micros());
        } else {
            self.frame_times.pop_front();
            self.frame_times.push_back(stats.delta.as_micros());
        }

        if self.cmd_buf_times.len() < 20 {
            self.cmd_buf_times.push_back(stats.cmd_buf_record_time.as_micros());
        } else {
            self.cmd_buf_times.pop_front();
            self.cmd_buf_times.push_back(stats.cmd_buf_record_time.as_micros());
        }
    }

    fn get_framerate(&self) -> f32 {
        let sum: u128 = self.frame_times.iter().sum();
        1.0 / ((sum as f32 / 1_000_000.0) / self.frame_times.len() as f32)
    }

    fn get_frame_time(&self) -> f32 {
        let sum: u128 = self.frame_times.iter().sum();
        sum as f32 / self.frame_times.len() as f32
    }

    fn get_cmd_buf_record_time(&self) -> f32 {
        let sum: u128 = self.cmd_buf_times.iter().sum();
        sum as f32 / self.cmd_buf_times.len() as f32
    }
}

#[derive(Debug, Clone, Copy)]
struct ShadowPass;

impl Task for ShadowPass {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: &DrawData, data: &mut RendererData, stats: &mut RenderStats) {
        begin_command_label(instance, device, command_buffer, "Shadow Pass", vec4(0.0, 0.6, 0.2, 1.0));
        
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }  })
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image_view(draw_data.depth_attachment.unwrap().view());

        let rendering_info = vk::RenderingInfo::default()
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: draw_data.depth_attachment.unwrap().extent_2d(),
            })
            .depth_attachment(&depth_attachment);

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: draw_data.depth_attachment.unwrap().extent_2d(),
        }];
    
        let viewports = [
            vk::Viewport::default()
                .width(draw_data.depth_attachment.unwrap().width() as f32)
                .height(draw_data.depth_attachment.unwrap().height() as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .x(0.0)
                .y(0.0)
        ];

        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        unsafe { device.cmd_set_viewport(command_buffer, 0, &viewports) };
        unsafe { device.cmd_set_scissor(command_buffer, 0, &scissors) };

        let (pipeline, pipeline_layout) = data.pipelines.get_with_tag("shadow")[0];
        unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline) };

        let light = data.objects.get_with_tag("light")[0];
        unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 0, &[*light.get_descriptor_set("camera_data")], &[]) };

        data.objects.get_with_tag("shadow").iter().for_each(|object| {
            unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 1, &[*object.get_descriptor_set("mvp")], &[]) };
            
            object.mesh_data().bind_buffers(device, command_buffer);
            unsafe { device.cmd_draw_indexed(command_buffer, object.mesh_data().index_len(), object.mesh_data().instance_len(), 0, 0, 0) };
            
            stats.draw_calls += 1;
        });

        unsafe { device.cmd_end_rendering(command_buffer) };
        
        end_command_label(instance, device, command_buffer);
    }

    fn recreate_swapchain(&mut self, _: &Instance, _: &Device, _: &mut RendererData, _: &DrawData) {}

    fn destroy(&mut self, _: &Device) {}
}

#[derive(Debug, Clone, Copy)]
struct MainDraw;

impl Task for MainDraw {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: &DrawData, data: &mut RendererData, stats: &mut RenderStats) {
        begin_command_label(instance, device, command_buffer, "Main Draw", vec4(0.0, 0.6, 0.2, 1.0));

        let attachment_infos = [
            vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0]}})
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image_view(draw_data.color_attachments[0].view())
        ];

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }  })
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image_view(draw_data.depth_attachment.unwrap().view());

        let rendering_info = vk::RenderingInfo::default()
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: draw_data.color_attachments[0].extent_2d(),
            })
            .color_attachments(&attachment_infos)
            .depth_attachment(&depth_attachment);

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: draw_data.color_attachments[0].extent_2d(),
        }];
    
        let viewports = [
            vk::Viewport::default()
                .width(draw_data.color_attachments[0].width() as f32)
                .height(draw_data.color_attachments[0].height() as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .x(0.0)
                .y(0.0)
        ];
        
        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        unsafe { device.cmd_set_viewport(command_buffer, 0, &viewports) };
        unsafe { device.cmd_set_scissor(command_buffer, 0, &scissors) };
        let (pipeline, pipeline_layout) = data.pipelines.get_with_tag("main")[0];
        unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline) };

        let camera = data.objects.get_with_tag("main_camera")[0];
        unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 0, &[*camera.get_descriptor_set("camera_data")], &[]) };

        let light = data.objects.get_with_tag("light")[0];
        unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 5, &[*light.get_descriptor_set("shadow_map")], &[]) };
        unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 2, &[*light.get_descriptor_set("camera_data")], &[]) };
        unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 3, &[*light.get_descriptor_set("light_data")], &[]) };

        data.objects.get_with_tag("main").iter().for_each(|object| {
            unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 1, &[*object.get_descriptor_set("mvp")], &[]) };
            unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 4, &[*object.get_descriptor_set("material")], &[]) };
            
            object.mesh_data().bind_buffers(device, command_buffer);
            unsafe { device.cmd_draw_indexed(command_buffer, object.mesh_data().index_len(), object.mesh_data().instance_len(), 0, 0, 0) };
            
            stats.draw_calls += 1;
        });

        unsafe { device.cmd_end_rendering(command_buffer) };

        end_command_label(instance, device, command_buffer);
    }

    fn recreate_swapchain(&mut self, _: &Instance, _: &Device, _: &mut RendererData, _: &DrawData) {}

    fn destroy(&mut self, _: &Device) {}
}

#[derive(Debug, Clone, Copy)]
struct ColourCorrectionPass;

impl Task for ColourCorrectionPass {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: &DrawData, data: &mut RendererData, stats: &mut RenderStats) {
        begin_command_label(instance, device, command_buffer, "Colour Correction", vec4(0.0, 0.5, 0.5, 1.0));

        let attachment_infos = [
            vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0]}})
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image_view(draw_data.color_attachments[0].view())
        ];

        let rendering_info = vk::RenderingInfo::default()
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: draw_data.color_attachments[0].extent_2d(),
            })
            .color_attachments(&attachment_infos);

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: draw_data.color_attachments[0].extent_2d(),
        }];
    
        let viewports = [
            vk::Viewport::default()
                .width(draw_data.color_attachments[0].width() as f32)
                .height(draw_data.color_attachments[0].height() as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .x(0.0)
                .y(0.0)
        ];
        
        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        unsafe { device.cmd_set_viewport(command_buffer, 0, &viewports) };
        unsafe { device.cmd_set_scissor(command_buffer, 0, &scissors) };

        let (pipeline, pipeline_layout) = data.pipelines.get_with_tag("colour_correction")[0];
        unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline) };

        let object = data.objects.get_with_tag("colour_correction")[0];
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *pipeline_layout,
                0,
                &[
                    *object.get_descriptor_set("colour_correction_data"),
                    *object.get_descriptor_set("colour_correction_images"),
                ],
                &[],
            ) 
        };

        unsafe { device.cmd_draw(command_buffer, 3, 1, 0, 0) };

        stats.draw_calls += 1;

        unsafe { device.cmd_end_rendering(command_buffer) };

        end_command_label(instance, device, command_buffer);
    }

    fn recreate_swapchain(&mut self, _: &Instance, device: &Device, data: &mut RendererData, draw_data: &DrawData) {
        let image = draw_data.internal_images[0];

        let cc_objects = data.objects.get_mut_with_tag("colour_correction");

        let descriptor_set = cc_objects[0].get_descriptor_set("colour_correction_images");
        update_image_binding(
            device,
            &[
                image.descriptor_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            ],
            *descriptor_set,
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
    }

    fn destroy(&mut self, _: &Device) {}
}

#[derive(Debug, Clone, Copy)]
struct EguiDraw;

impl Task for EguiDraw {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: &DrawData, data: &mut RendererData, _: &mut RenderStats) {
        let enable = true;

        if !enable { return; }

        begin_command_label(instance, device, command_buffer, "Egui Draw", vec4(0.4, 0.5, 0.7, 1.0));
        
        let attachment_infos = [
            vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.05, 0.5, 0.0, 1.0]}})
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image_view(draw_data.color_attachments[0].view())
        ];

        let rendering_info = vk::RenderingInfo::default()
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: draw_data.color_attachments[0].extent_2d(),
            })
            .color_attachments(&attachment_infos);
        
        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        data.egui_renderer.as_mut().unwrap().cmd_draw(command_buffer, draw_data.color_attachments[0].extent_2d(), data.pixels_per_point, &data.clipped_primitives).unwrap();

        unsafe { device.cmd_end_rendering(command_buffer) };

        end_command_label(instance, device, command_buffer);
    }

    fn recreate_swapchain(&mut self, _: &ash::Instance, _: &Device, _: &mut RendererData, _: &DrawData) {}

    fn destroy(&mut self, _: &Device) {}
}

#[derive(Debug, Clone, Copy)]
struct DebugDraw;

impl Task for DebugDraw {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: &DrawData, data: &mut RendererData, stats: &mut RenderStats) {
        begin_command_label(instance, device, command_buffer, "Debug Draw", vec4(0.0, 0.6, 0.2, 1.0));

        let attachment_infos = [
            vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.01, 0.01, 0.01, 1.0]}})
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image_view(draw_data.color_attachments[0].view())
        ];

        let rendering_info = vk::RenderingInfo::default()
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: draw_data.color_attachments[0].extent_2d(),
            })
            .color_attachments(&attachment_infos);

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: draw_data.color_attachments[0].extent_2d(),
        }];
    
        let viewports = [
            vk::Viewport::default()
                .width(draw_data.color_attachments[0].width() as f32)
                .height(draw_data.color_attachments[0].height() as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .x(0.0)
                .y(0.0)
        ];
        
        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        unsafe { device.cmd_set_viewport(command_buffer, 0, &viewports) };
        unsafe { device.cmd_set_scissor(command_buffer, 0, &scissors) };
        let (pipeline, pipeline_layout) = data.pipelines.get_with_tag("debug")[0];
        unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline) };

        let camera = data.objects.get_with_tag("main_camera")[0];
        unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 0, &[*camera.get_descriptor_set("camera_data")], &[]) };

        data.objects.get_with_tag("debug").iter().for_each(|object| {
            unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 1, &[*object.get_descriptor_set("mvp")], &[]) };

            object.mesh_data().bind_buffers(device, command_buffer);
            unsafe { device.cmd_draw(command_buffer, object.mesh_data().vertex_len(), object.mesh_data().instance_len(), 0, 0) };

            stats.draw_calls += 1;
        });

        unsafe { device.cmd_end_rendering(command_buffer) };

        end_command_label(instance, device, command_buffer);
    }

    fn recreate_swapchain(&mut self, _: &Instance, _: &Device, _: &mut RendererData, _: &DrawData) {}

    fn destroy(&mut self, _: &Device) {}
}

#[derive(Debug, Clone, Copy)]
struct Present;

impl Task for Present {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: &DrawData, data: &mut RendererData, _: &mut RenderStats) {
        begin_command_label(instance, device, command_buffer, "Present", vec4(0.7, 0.0, 0.7, 1.0));
        
        transition_image_access(device, command_buffer, data.swapchain_images[data.image_index], AccessType::UNDEFINED, AccessType::transfer_dst(), vk::ImageAspectFlags::COLOR);

        copy_image_to_image(device, command_buffer, draw_data.color_attachments[0].image(), data.swapchain_images[data.image_index], draw_data.color_attachments[0].extent_2d(), data.swapchain_extent, vk::Filter::NEAREST);

        transition_image_access(device, command_buffer, data.swapchain_images[data.image_index], AccessType::transfer_dst(), AccessType::present_src(), vk::ImageAspectFlags::COLOR);
        end_command_label(instance, device, command_buffer);
    }
    
    fn recreate_swapchain(&mut self, _: &ash::Instance, _: &Device, _: &mut RendererData, _: &DrawData) {}

    fn destroy(&mut self, _: &Device) {}
}

#[repr(C)]
struct MaterialData {
    specular: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct ColourCorrection {
    levels: Levels,
    hue_shift: f32,
}

#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Levels {
    in_black: f32,
    in_white: f32,
    out_black: f32,
    out_white: f32,
    gamma: f32,
}

#[derive(Vertex)]
pub struct NullVertex;
