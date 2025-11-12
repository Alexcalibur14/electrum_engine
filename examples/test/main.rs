use std::collections::VecDeque;

use anyhow::Result;
use electrum_engine::{RenderStats, Renderer};
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
struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,

    egui_data: EguiData,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(Window::default_attributes()
            .with_title("Electrum Renderer Example")).unwrap();

        let renderer = Renderer::new(&window).unwrap();

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
