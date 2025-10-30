use anyhow::Result;
use electrum_engine::Renderer;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::{ControlFlow, EventLoopBuilder}, window::Window};

fn main() -> Result<()> {
    let layer = tracing_subscriber::fmt::layer().with_filter(LevelFilter::INFO);
    tracing_subscriber::registry().with(layer).init();

    info!("Starting..");
    let event_loop = EventLoopBuilder::default()
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(Window::default_attributes()
            .with_title("Electrum Renderer Example")).unwrap();

        let renderer = Renderer::new(&window).unwrap();

        self.window = Some(window);
        self.renderer = Some(renderer)
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                if event_loop.exiting() {
                    return;
                }
                let renderer = self.renderer.as_mut().unwrap();
                let window = self.window.as_ref().unwrap();

                unsafe { renderer.render(window) }.unwrap();

                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}
