use std::fs;
use std::io::Read;
use std::fs::File;

use anyhow::Result;
use shaderc::{CompileOptions, Compiler, ShaderKind};

use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_2::*;

use crate::{insert_command_label, set_object_name, Loadable, MeshData, RendererData};

pub trait Shader {
    fn stages(&self) -> Vec<vk::PipelineShaderStageCreateInfo>;
    fn destroy(&self, device: &Device);
    fn clone_dyn(&self) -> Box<dyn Shader>;
    fn loaded(&self) -> bool;
}

impl Clone for Box<dyn Shader> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

impl Loadable for Box<dyn Shader> {
    fn is_loaded(&self) -> bool {
        self.loaded()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VFShader {
    pub vertex: vk::ShaderModule,
    pub fragment: vk::ShaderModule,
    loaded: bool,
}

impl VFShader {
    pub fn builder(instance: &Instance, device: &Device, name: String) -> VFShaderBuilder {
        VFShaderBuilder::new(instance.clone(), device.clone(), name)
    }
}

impl Shader for VFShader {
    fn stages(&self) -> Vec<vk::PipelineShaderStageCreateInfo> {
        vec![
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vertex)
                .name(b"main\0")
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fragment)
                .name(b"main\0")
                .build(),
        ]
    }

    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.vertex, None);
            device.destroy_shader_module(self.fragment, None);
        }
    }

    fn clone_dyn(&self) -> Box<dyn Shader> {
        Box::new(*self)
    }
    
    fn loaded(&self) -> bool {
        self.loaded
    }
    
}

pub struct VFShaderBuilder {
    instance: Instance,
    device: Device,
    name: String,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
}

impl VFShaderBuilder {
    fn new(instance: Instance, device: Device, name: String) -> Self {
        VFShaderBuilder {
            instance,
            device,
            name,
            vertex: vk::ShaderModule::default(),
            fragment: vk::ShaderModule::default(),
        }
    }

    /// Compiles the vertex shader at the location from `path`
    pub fn compile_vertex(&mut self, path: &str) -> &mut Self {
        let vertex =
            compile_shader_module(&self.device, path, "main", ShaderKind::Vertex)
                .unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            self.name.clone() + " Vertex Shader",
            vk::ObjectType::SHADER_MODULE,
            vertex.as_raw(),
        )
        .unwrap();

        self.vertex = vertex;
        self
    }

    pub fn load_vertex(&mut self, path: &str) -> &mut Self {
        let bytecode = Bytecode::new(&fs::read(path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code_size(bytecode.code_size())
            .code(bytecode.code())
            .build();

        let vertex = unsafe { self.device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            self.name.clone() + " Vertex Shader",
            vk::ObjectType::SHADER_MODULE,
            vertex.as_raw(),
        )
        .unwrap();

        self.vertex = vertex;
        self
    }

    /// Compiles the fragment shader at the location from `path`
    pub fn compile_fragment(&mut self, path: &str) -> &mut Self {
        let fragment =
            compile_shader_module(&self.device, path, "main", ShaderKind::Fragment)
                .unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            self.name.clone() + " Fragment Shader",
            vk::ObjectType::SHADER_MODULE,
            fragment.as_raw(),
        )
        .unwrap();

        self.fragment = fragment;
        self
    }

    pub fn load_fragment(&mut self, path: &str) -> &mut Self {
        let bytecode = Bytecode::new(&fs::read(path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code_size(bytecode.code_size())
            .code(bytecode.code())
            .build();

        let fragment = unsafe { self.device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            self.name.clone() + " Fragment Shader",
            vk::ObjectType::SHADER_MODULE,
            fragment.as_raw(),
        )
        .unwrap();

        self.fragment = fragment;
        self
    }

    pub fn build(&mut self) -> VFShader {
        VFShader {
            vertex: self.vertex,
            fragment: self.fragment,
            loaded: true,
        }
    }
}

/// Compiles a shader file
pub fn compile_shader_module(
    device: &Device,
    path: &str,
    entry_point_name: &str,
    shader_kind: ShaderKind,
) -> Result<vk::ShaderModule> {
    let mut f = File::open(path).unwrap_or_else(|_| panic!("Could not open file {}", path));
    let mut glsl = String::new();

    f.read_to_string(&mut glsl)
        .unwrap_or_else(|_| panic!("Could not read file {} to string", path));

    let compiler = Compiler::new().unwrap();
    let mut options = CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some(entry_point_name));

    let compiled = compiler
        .compile_into_spirv(&glsl, shader_kind, path, entry_point_name, Some(&options))
        .expect("Could not compile glsl shader to spriv");

    let bytes = compiled.as_binary_u8();

    let bytecode = Bytecode::new(bytes).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(unsafe { device.create_shader_module(&info, None) }?)
}

#[derive(Debug, Clone)]
pub struct Material {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,

    pub push_constant_ranges: Vec<vk::PushConstantRange>,

    shader: usize,
    mesh_settings: PipelineMeshSettings,

    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub other_set_layouts: Vec<vk::DescriptorSetLayout>,

    loaded: bool,
    subpass: u32,
}

impl Material {
    pub fn new(
        device: &Device,
        data: &mut RendererData,
        bindings: Vec<vk::DescriptorSetLayoutBinding>,
        push_constant_sizes: Vec<(u32, vk::ShaderStageFlags)>,
        shader: usize,
        mesh_settings: PipelineMeshSettings,
        other_layouts: Vec<vk::DescriptorSetLayout>,
        subpass: u32,
    ) -> Self {
        let mut push_constant_ranges = vec![];
        let mut offset = 0u32;
        for (size, stage_flag) in &push_constant_sizes {
            let range = vk::PushConstantRange::builder()
                .stage_flags(*stage_flag)
                .offset(offset)
                .size(*size)
                .build();

            offset += size;
            push_constant_ranges.push(range);
        }

        let descriptor_set_layout = data.global_layout_cache.create_descriptor_set_layout(device, &bindings);

        let mut set_layouts = vec![descriptor_set_layout];
        set_layouts.append(&mut other_layouts.clone());

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges)
            .build();

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }.unwrap();

        let pipeline = unsafe {
            create_pipeline(device, data, shader, mesh_settings.clone(), pipeline_layout, subpass)
        }
        .unwrap();

        Material {
            pipeline,
            pipeline_layout,

            push_constant_ranges,

            shader,
            mesh_settings,

            descriptor_set_layout,
            other_set_layouts: other_layouts.clone(),
            loaded: true,
            subpass,
        }
    }

    pub fn destroy_swapchain(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }

    pub fn recreate_swapchain(&mut self, device: &Device, data: &RendererData) {
        let mut set_layouts = vec![self.descriptor_set_layout];

        set_layouts.append(&mut self.other_set_layouts.clone());

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&self.push_constant_ranges)
            .build();

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }.unwrap();

        self.pipeline = unsafe {
            create_pipeline(
                device,
                data,
                self.shader,
                self.mesh_settings.clone(),
                pipeline_layout,
                self.subpass,
            )
        }
        .unwrap();

        self.pipeline_layout = pipeline_layout;
    }

    pub fn draw(
        &self,
        instance: &Instance,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        other_descriptors: Vec<(u32, vk::DescriptorSet)>,
        mesh_data: &MeshData,
        name: &str,
    ) {
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            for (set, descriptor) in other_descriptors {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    set,
                    &[descriptor],
                    &[],
                );
            }
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[mesh_data.vertex_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(
                command_buffer,
                mesh_data.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );
            insert_command_label(
                instance,
                command_buffer,
                format!("Draw {}", name),
                [0.0, 0.5, 0.1, 1.0],
            );
            device.cmd_draw_indexed(command_buffer, mesh_data.index_count, 1, 0, 0, 0);
        }
    }
}

impl Loadable for Material {
    fn is_loaded(&self) -> bool {
        self.loaded
    }
}

unsafe fn create_pipeline(
    device: &Device,
    data: &RendererData,
    shader: usize,
    mesh_settings: PipelineMeshSettings,
    pipeline_layout: vk::PipelineLayout,
    subpass: u32,
) -> Result<vk::Pipeline> {
    // Vertex Input State

    let binding_descriptions = mesh_settings.binding_descriptions;
    let attribute_descriptions = mesh_settings.attribute_descriptions;
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    // Input Assembly State

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(mesh_settings.topology)
        .primitive_restart_enable(false);

    // Viewport State

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    // Rasterization State

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(mesh_settings.polygon_mode)
        .line_width(mesh_settings.line_width)
        .cull_mode(mesh_settings.cull_mode)
        .front_face(mesh_settings.front_face)
        .depth_bias_enable(false);

    // Multisample State

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(data.msaa_samples);

    // Color Blend State

    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // Depth state

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // Create

    let shader_stages = data.shaders.get(shader).unwrap().stages();

    let stages = shader_stages.as_slice();
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(subpass);

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    Ok(pipeline)
}

#[derive(Debug, Clone)]
pub struct PipelineMeshSettings {
    pub binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,

    pub topology: vk::PrimitiveTopology,

    pub polygon_mode: vk::PolygonMode,
    pub line_width: f32,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
}

impl Default for PipelineMeshSettings {
    fn default() -> Self {
        PipelineMeshSettings {
            binding_descriptions: vec![],
            attribute_descriptions: vec![],

            topology: vk::PrimitiveTopology::TRIANGLE_LIST,

            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }
}
