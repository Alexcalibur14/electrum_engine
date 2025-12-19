use std::collections::VecDeque;

use ash::Device;
use ash::Instance;
use ash::vk;

use anyhow::Result;
use electrum_engine::begin_command_label;
use electrum_engine::buffer::Buffer;
use electrum_engine::descriptor::DescriptorBuilder;
use electrum_engine::end_command_label;
use electrum_engine::image::MipLevels;
use electrum_engine::image::copy_image_to_image;
use electrum_engine::model::OBJVertex;
use electrum_engine::model::Object;
use electrum_engine::model::basic_obj_loader;
use electrum_engine::shader::DepthStencilData;
use electrum_engine::shader::GraphicsProgram;
use electrum_engine::shader::MultisampleData;
use electrum_engine::shader::RasterizationData;
use electrum_engine::shader::create_basic_graphics_pipeline;
use electrum_engine::task_graph::*;
use electrum_engine::{RenderStats, Renderer, RendererData};
use glam::Quat;
use glam::vec3;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
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
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(Window::default_attributes()
            .with_title("Electrum Renderer Example")
            .with_resizable(true)
        ).unwrap();

        let mut renderer = Renderer::new(&window).unwrap();

        let mut program = GraphicsProgram::new("test", "res/shaders/test.vert.spv");
        program.set_fragment_path("res/shaders/test.frag.spv");
        program.load_shader_modules_spirv(&renderer.instance, &renderer.device);

        let matrix = glam::Mat4::from_scale_rotation_translation(vec3(0.8, -0.8, 0.8), Quat::IDENTITY, vec3(0.0, 0.0, 0.0));
        let object_buffer = Buffer::create_and_stage(&renderer.instance, &renderer.device, &renderer.data, &matrix.to_cols_array(), vk::BufferUsageFlags::UNIFORM_BUFFER, "object_matrix");

        let buffer_info = &[
            vk::DescriptorBufferInfo::default()
                .buffer(object_buffer.buffer())
                .offset(0)
                .range(object_buffer.size())
        ];

        let (object_descriptor, layout) = DescriptorBuilder::new()
            .bind_buffer(0, 1, buffer_info, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
            .build_data(&renderer.device, &mut renderer.data).unwrap();

        let mut monkey = Object::new("monkey");
        monkey.add_buffer(object_buffer, "mvp");
        monkey.add_descriptor_set(object_descriptor, "mvp");

        *monkey.mesh_data_mut() = basic_obj_loader(&renderer.instance, &renderer.device, &renderer.data, "res/models/MONKEY.obj")[0].clone();

        renderer.data.objects.push(monkey, &["main"]);

        let (pipeline, layout) = create_basic_graphics_pipeline::<OBJVertex>(
            &renderer.instance,
            &renderer.device,
            &program,
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
            &[layout],
            vk::PrimitiveTopology::TRIANGLE_LIST
        );

        renderer.data.graphics_shaders.push(program, &["main"]);
        renderer.data.pipelines.push((pipeline, layout), &["main"]);

        let mut task_graph = TaskGraph::new();

        let image = ImageData::new(
            ImageSize::SwapchainRelative { multiplier: 1.0 },
            vk::Format::R16G16B16A16_SFLOAT,
            MipLevels::One,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageViewType::TYPE_2D,
            1,
            vk::ImageAspectFlags::COLOR,
            None,
            "color_attachment"
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

        task_graph.add_image(image);
        task_graph.add_image(depth);

        let mut main_node = Node::new();
        main_node.add_attachment("color_attachment", AccessType::color_attachment_write());
        main_node.set_depth("depth_attachment", AccessType::depth_stencil_attachment_read() | AccessType::depth_stencil_attachment_write());
        main_node.set_task(Box::new(MainDraw));
        task_graph.add_node(main_node);

        let mut egui_node = Node::new();
        egui_node.add_attachment("color_attachment", AccessType::color_attachment_write());
        egui_node.set_task(Box::new(EguiDraw));
        task_graph.add_node(egui_node);

        let mut present_node = Node::new();
        present_node.add_attachment("color_attachment", AccessType::blit_src());
        present_node.set_task(Box::new(Present));
        task_graph.add_node(present_node);

        task_graph.create_images(&renderer.instance, &renderer.device, &renderer.data);
        renderer.data.task_graph = task_graph;


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

                unsafe { renderer.render(window, &mut |ctx| {
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
                }) }.unwrap();

                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(new_size) => {
                let renderer = self.renderer.as_mut().unwrap();
                renderer.resized = true;
                renderer.width = new_size.width;
                renderer.height = new_size.height;
            }
            _ => (),
        }
    }
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
struct MainDraw;

impl Task for MainDraw {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: DrawData<'a>, data: &mut RendererData, stats: &mut RenderStats) {
        begin_command_label(instance, device, command_buffer, "Main Draw", [0.0, 0.6, 0.2, 1.0]);

        let attachment_infos = [
            vk::RenderingAttachmentInfo::default()
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.05, 0.5, 0.0, 1.0]}})
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

        
        data.objects.get_with_tag("main").iter().for_each(|object| {
            unsafe { device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 0, &[*object.get_descriptor_set("mvp")], &[]) };
            
            object.mesh_data().bind_buffers(device, command_buffer);
            unsafe { device.cmd_draw_indexed(command_buffer, object.mesh_data().index_len(), object.mesh_data().instance_len(), 0, 0, 0) };
        });

        stats.draw_calls += 1;
        
        unsafe { device.cmd_end_rendering(command_buffer) };

        end_command_label(instance, device, command_buffer);
    }

    fn recreate_swapchain(&mut self, _: &ash::Instance, _: &Device, _: &RendererData) {}

    fn destroy(&mut self, _: &Device) {}
}

#[derive(Debug, Clone, Copy)]
struct EguiDraw;

impl Task for EguiDraw {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: DrawData<'a>, data: &mut RendererData, stats: &mut RenderStats) {
        begin_command_label(instance, device, command_buffer, "Egui Draw", [0.4, 0.5, 0.7, 1.0]);
        
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

        stats.draw_calls += 1;

        unsafe { device.cmd_end_rendering(command_buffer) };

        end_command_label(instance, device, command_buffer);
    }

    fn recreate_swapchain(&mut self, _: &ash::Instance, _: &Device, _: &RendererData) {}

    fn destroy(&mut self, _: &Device) {}
}

#[derive(Debug, Clone, Copy)]
struct Present;

impl Task for Present {
    fn execute<'a>(&self, instance: &Instance, device: &Device, command_buffer: vk::CommandBuffer, draw_data: DrawData<'a>, data: &mut RendererData, _: &mut RenderStats) {
        begin_command_label(instance, device, command_buffer, "Present", [0.7, 0.0, 0.7, 1.0]);
        
        transition_image_access(device, command_buffer, data.swapchain_images[data.image_index], AccessType::UNDEFINED, AccessType::transfer_dst(), vk::ImageAspectFlags::COLOR);

        copy_image_to_image(device, command_buffer, draw_data.color_attachments[0].image(), data.swapchain_images[data.image_index], draw_data.color_attachments[0].extent_2d(), data.swapchain_extent, vk::Filter::NEAREST);

        transition_image_access(device, command_buffer, data.swapchain_images[data.image_index], AccessType::transfer_dst(), AccessType::present_src(), vk::ImageAspectFlags::COLOR);
        end_command_label(instance, device, command_buffer);
    }
    
    fn recreate_swapchain(&mut self, _: &ash::Instance, _: &Device, _: &RendererData) {}

    fn destroy(&mut self, _: &Device) {}
}
