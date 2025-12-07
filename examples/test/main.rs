use std::collections::VecDeque;

use ash::Device;
use ash::vk;

use anyhow::Result;
use electrum_engine::TestVertex;
use electrum_engine::buffer::create_and_stage_buffer;
use electrum_engine::create_pipeline;
use electrum_engine::draw::bind_index_vertex;
use electrum_engine::image::Image;
use electrum_engine::image::MipLevels;
use electrum_engine::present::setup_and_copy_to_swapchain;
use electrum_engine::resources::Handle;
use electrum_engine::{RenderStats, Renderer, RendererData, present::transition_image};
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::{ControlFlow, EventLoopBuilder}, platform::x11::EventLoopBuilderExtX11, window::Window};

fn main() -> Result<()> {
    let layer = tracing_subscriber::fmt::layer().with_filter(LevelFilter::INFO);
    tracing_subscriber::registry().with(layer).init();

    info!("Starting..");
    let event_loop = EventLoopBuilder::default()
        .with_x11() // egui selection is offset from cursor on wayland
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
            .with_title("Electrum Renderer Example")).unwrap();

        let mut renderer = Renderer::new(&window).unwrap();

        renderer.data.draw_func = draw;

        let (width, height) = {
            let inner_size = window.inner_size();
            (inner_size.width, inner_size.height)
        };

        let pipeline = create_pipeline(&renderer.device, width, height);
        renderer.data.pipeline = pipeline;

        let vertices = create_and_stage_buffer(&renderer.instance, &renderer.device, &renderer.data, (std::mem::size_of::<TestVertex>() * 6) as u64, vk::BufferUsageFlags::VERTEX_BUFFER, "Vertex Buffer", &[
            TestVertex{ position: [ 0.0 , -0.5 ], colour: [0.0 , 1.0 , 0.0] },
            TestVertex{ position: [-0.25, -0.25], colour: [0.25, 0.75, 0.25] },
            TestVertex{ position: [ 0.25, -0.25], colour: [0.25, 0.75, 0.25] },
            TestVertex{ position: [-0.25,  0.25], colour: [0.75, 0.25, 0.75] },
            TestVertex{ position: [ 0.25,  0.25], colour: [0.75, 0.25, 0.75] },
            TestVertex{ position: [ 0.0 ,  0.5 ], colour: [1.0 , 0.0 , 1.0] },
        ]).unwrap();
        renderer.data.vertices = vertices;

        let indices = create_and_stage_buffer(&renderer.instance, &renderer.device, &renderer.data, (std::mem::size_of::<u32>() * 12) as u64, vk::BufferUsageFlags::INDEX_BUFFER, "Index Buffer", &[
            0, 1, 2,
            1, 3, 2,
            2, 3, 4,
            3, 5, 4,
        ]).unwrap();
        renderer.data.indices = indices;

        let colour_attachement = Image::new(
            &renderer.instance,
            &renderer.device,
            &renderer.data,
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
            false,
            "color_attachment",
        );
        renderer.data.attachments.push(colour_attachement, &[]);


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
                println!("The close button was pressed; stopping");
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
                            ui.label(format!("Frame Time: {:.2} ms", self.egui_data.get_frame_time()));
                            ui.label(format!("Frame number: {}", stats.frame));
                            ui.label(format!("Draw Calls: {}", stats.draw_calls));
                            ui.label(format!("Command Buffer time: {:.2} us", self.egui_data.get_cmd_buf_record_time()))
                    });
                }) }.unwrap();

                self.window.as_ref().unwrap().request_redraw();
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
        if self.frame_times.len() < 10 {
            self.frame_times.push_back(stats.delta.as_millis());
        } else {
            self.frame_times.pop_front();
            self.frame_times.push_back(stats.delta.as_millis());
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
        1.0 / ((sum as f32 / 1000.0) / self.frame_times.len() as f32)
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

fn draw(device: &Device, command_buffer: vk::CommandBuffer, image_index: usize, data: &mut RendererData, stats: &mut RenderStats, clipped_primitives: &[egui::ClippedPrimitive], pixels_per_point: f32) {
    let attachment_handle = Handle::new(0);
    
    let attachment_infos = [
        vk::RenderingAttachmentInfo::default()
        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.05, 0.5, 0.0, 1.0]}})
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .image_view(data.attachments.get(&attachment_handle).unwrap().view)
    ];

    let rendering_info = vk::RenderingInfo::default()
        .layer_count(1)
        .render_area(data.swapchain_rect)
        .color_attachments(&attachment_infos);

    transition_image(device, command_buffer, data.attachments.get(&attachment_handle).unwrap().image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

    bind_index_vertex(device, command_buffer, data.indices.buffer, vk::IndexType::UINT32, data.vertices.buffer);
    unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, data.pipeline) };
    unsafe { device.cmd_draw_indexed(command_buffer, 12, 1, 0, 0, 0) };

    stats.draw_calls += 1;

    data.egui_renderer.as_mut().unwrap().cmd_draw(command_buffer, data.swapchain_extent, pixels_per_point, clipped_primitives).unwrap();

    stats.draw_calls += 1;

    unsafe { device.cmd_end_rendering(command_buffer) };

    setup_and_copy_to_swapchain(device, data, command_buffer, data.swapchain_images[image_index], data.attachments.get(&attachment_handle).unwrap().image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
}
