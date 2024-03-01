use std::fs::File;
use std::io::Read;

use shaderc::{CompileOptions, Compiler, ShaderKind};
use anyhow::Result;

use vulkanalia::prelude::v1_2::*;
use vulkanalia::bytecode::Bytecode;

use crate::model::Vertex;
use crate::AppData;

pub trait Shader {
    fn stages(&self) -> Vec<vk::PipelineShaderStageCreateInfo>;
}


#[derive(Debug, Clone, Copy, Default)]
pub struct VFShader {
    pub vertex: vk::ShaderModule,
    pub fragment: vk::ShaderModule,
}

impl VFShader {
    /// Compiles the vertex shader at the location from `path`
    fn compile_vertex(&mut self, device: &Device, path: &str) {
        self.vertex = unsafe { compile_shader_module(&device, path, "main", ShaderKind::Vertex) }.unwrap();
    }

    /// Compiles the fragment shader at the location from `path`
    fn compile_fragment(&mut self, device: &Device, path: &str) {
        self.fragment = unsafe { compile_shader_module(&device, path, "main", ShaderKind::Fragment) }.unwrap();
    }

    unsafe fn destroy(&self, device: &Device) {
        device.destroy_shader_module(self.vertex, None);
        device.destroy_shader_module(self.fragment, None);
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
}

/// Compiles a shader file
pub unsafe fn compile_shader_module(device: &Device, path: &str, entry_point_name: &str, shader_kind: ShaderKind) -> Result<vk::ShaderModule> {
    let mut f = File::open(path).unwrap_or_else(|_| panic!("Could not open file {}", path));
    let mut glsl = String::new();

    f.read_to_string(&mut glsl).unwrap_or_else(|_| panic!("Could not read file {} to string", path));

    let compiler = Compiler::new().unwrap();
    let mut options = CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some(entry_point_name));

    let binding = compiler.compile_into_spirv(&glsl, shader_kind, path, entry_point_name, Some(&options)).expect("Could not compile glsl shader to spriv");
    let bytes = binding.as_binary_u8();

    let bytecode = Bytecode::new(&bytes[..]).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}


#[derive(Debug, Default, Clone)]
pub struct Material {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,

    pub bindings: Vec<vk::DescriptorSetLayoutBinding>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,

    pub push_constant_sizes: Vec<(u32, vk::ShaderStageFlags)>,
}

impl Material {
    pub fn new(
        device: &Device,
        data: &AppData,
        bindings: Vec<vk::DescriptorSetLayoutBinding>,
        push_constant_sizes: Vec<(u32, vk::ShaderStageFlags)>,
        shader: &impl Shader,
        vertex: &impl Vertex,
    ) -> Self {
        
        
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(bindings.as_slice())
            .build();

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None) }.unwrap();

        let descriptor_pool = unsafe { create_descriptor_pool(device, data, &bindings) }.unwrap();

        let (pipeline, pipeline_layout) = unsafe { create_pipeline(device, data, shader, push_constant_sizes.clone(), descriptor_set_layout, vertex) }.unwrap();

        Material {
            pipeline,
            pipeline_layout,
            descriptor_pool,

            bindings,
            descriptor_set_layout,

            push_constant_sizes,
        }
    }
}

unsafe fn create_pipeline(device: &Device, data: &AppData, shader: &impl Shader, push_constant_sizes: Vec<(u32, vk::ShaderStageFlags)>, descriptor_set_layout:vk::DescriptorSetLayout, vertex: &impl Vertex) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
    // Vertex Input State
    
    let binding_descriptions = &[vertex.binding_description()];
    let attribute_descriptions = vertex.attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    // Input Assembly State

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
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
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);

    // Multisample State

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(data.msaa_samples);


    // Color Blend State

    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

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

    // Layout

    let mut push_constant_ranges = vec![];
    let mut offset = 0u32;
    for (size, stage_flag) in &push_constant_sizes {
        let range = vk::PushConstantRange::builder()
            .stage_flags(*stage_flag)
            .offset(offset)
            .size(*size);

        offset += size;
        push_constant_ranges.push(range);
    }

    let set_layouts = &[descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges.as_slice());

    let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;


    // Create

    let shader_stages = shader.stages();

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
        .subpass(0);

    let pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?.0[0];

    Ok((pipeline, pipeline_layout))
}

unsafe fn create_descriptor_pool(device: &Device, data: &AppData, bindings: &Vec<vk::DescriptorSetLayoutBinding>) -> Result<vk::DescriptorPool> {
    let images = data.swapchain_images.len() as u32;

    let mut sizes = vec![];

    for binding in bindings {
        let size = vk::DescriptorPoolSize::builder()
            .type_(binding.descriptor_type)
            .descriptor_count(binding.descriptor_count * images)
            .build();

        sizes.push(size);
    }

    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(sizes.as_slice())
        .max_sets(data.swapchain_images.len() as u32);
    
    let descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(descriptor_pool)
}

