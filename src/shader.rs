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
    fn states(&self) -> PipelineShaderState {
        PipelineShaderState {
            stages: self.stages(),
            rasterisation: vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false)
                .build(),

            tessalation: vk::PipelineTessellationStateCreateInfo::builder()
                .build(),

            dynamic: vk::PipelineDynamicStateCreateInfo::builder()
                .build(),
        }
    }
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

    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub other_set_layouts: Vec<vk::DescriptorSetLayout>,

    pub mesh_state: PipelineMeshState,
    pub shader_state: PipelineShaderState,
    pub subpass_state: SubpassPipelineState,

    loaded: bool,
    subpass: u32,
}

impl Material {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,
        bindings: Vec<vk::DescriptorSetLayoutBinding>,
        push_constant_sizes: Vec<(u32, vk::ShaderStageFlags)>,
        shader_state: PipelineShaderState,
        subpass_state: SubpassPipelineState,
        mesh_state: PipelineMeshState,
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

        let pipeline = create_pipeline_from_states(instance, device, pipeline_layout, &subpass_state, &shader_state, &mesh_state, subpass, data.render_pass, "").unwrap();

        Material {
            pipeline,
            pipeline_layout,

            push_constant_ranges,

            mesh_state: mesh_state.clone(),
            shader_state: shader_state.clone(),
            subpass_state: subpass_state.clone(),

            descriptor_set_layout,
            other_set_layouts: other_layouts.clone(),
            
            subpass,
            loaded: true,
        }
    }

    pub fn destroy_swapchain(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }

    pub fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &RendererData) {
        let mut set_layouts = vec![self.descriptor_set_layout];

        set_layouts.append(&mut self.other_set_layouts.clone());

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&self.push_constant_ranges)
            .build();

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }.unwrap();

        self.pipeline = create_pipeline_from_states(
            instance,
            device,
            pipeline_layout,
            &self.subpass_state,
            &self.shader_state,
            &self.mesh_state,
            self.subpass,
            data.render_pass,
            "",
        )
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


#[derive(Debug, Clone)]
pub struct SubpassPipelineState {
    pub viewports: Vec<vk::Viewport>,
    pub scissors: Vec<vk::Rect2D>,

    pub rasterisation_samples: vk::SampleCountFlags,
 
    pub depth_test: bool,
    pub depth_write: bool,

    pub attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    pub logic_op_enable: bool,
    pub logic_op: vk::LogicOp,
    pub blend_constants: [f32; 4],
}

impl SubpassPipelineState {
    pub fn new(
        viewports: Vec<vk::Viewport>,
        scissors: Vec<vk::Rect2D>,

        rasterisation_samples: vk::SampleCountFlags,

        depth_test: bool,
        depth_write: bool,

        attachments: Vec<vk::PipelineColorBlendAttachmentState>,
        logic_op_enable: bool,
        logic_op: vk::LogicOp,
        blend_constants: [f32; 4],
    ) -> Self {
        SubpassPipelineState {
            viewports,
            scissors,
            rasterisation_samples,
            depth_test,
            depth_write,
            attachments,
            logic_op_enable,
            logic_op,
            blend_constants,
        }
    }

    pub fn viewport(&self) -> vk::PipelineViewportStateCreateInfo {
        vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&self.viewports)
            .scissors(&self.scissors)
            .build()
    }

    pub fn multisample(&self) -> vk::PipelineMultisampleStateCreateInfo {
        vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(self.rasterisation_samples)
            .sample_shading_enable(false)
            .build()
    }

    pub fn depth_stencil(&self) -> vk::PipelineDepthStencilStateCreateInfo {
        vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(self.depth_test)
            .depth_write_enable(self.depth_write)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .build()
    }

    pub fn color_blend(&self) -> vk::PipelineColorBlendStateCreateInfo {
        vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(self.logic_op_enable)
            .logic_op(self.logic_op)
            .attachments(&self.attachments)
            .blend_constants(self.blend_constants)
            .build()
    }
}


#[derive(Debug, Clone)]
pub struct PipelineShaderState {
    pub stages: Vec<vk::PipelineShaderStageCreateInfo>,
    pub rasterisation: vk::PipelineRasterizationStateCreateInfo,
    pub tessalation: vk::PipelineTessellationStateCreateInfo,
    pub dynamic: vk::PipelineDynamicStateCreateInfo,
}

#[derive(Debug, Clone)]
pub struct PipelineMeshState {
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub attributes: Vec<vk::VertexInputAttributeDescription>,
    pub primitive_restart: bool,
    pub topology: vk::PrimitiveTopology,
}

impl PipelineMeshState {
    pub fn new(
        binding_descriptions: Vec<vk::VertexInputBindingDescription>,
        attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
        primitive_restart: bool,
        topology: vk::PrimitiveTopology,
    ) -> Self {
        Self {
            bindings: binding_descriptions,
            attributes: attribute_descriptions,
            primitive_restart,
            topology,
        }
    }

    pub fn vertex_input(&self) -> vk::PipelineVertexInputStateCreateInfo {
        vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&self.bindings)
            .vertex_attribute_descriptions(&self.attributes)
            .build()
    }

    pub fn input_assembly(&self) -> vk::PipelineInputAssemblyStateCreateInfo {
        vk::PipelineInputAssemblyStateCreateInfo::builder()
            .primitive_restart_enable(self.primitive_restart)
            .topology(self.topology)
            .build()
    }
}


pub fn create_pipeline_from_states(
    instance: &Instance,
    device: &Device,
    layout: vk::PipelineLayout,
    subpass_state: &SubpassPipelineState,
    shader_state: &PipelineShaderState,
    mesh_state: &PipelineMeshState,
    subpass: u32,
    render_pass: vk::RenderPass,
    material_name: &str,
) -> Result<vk::Pipeline> {
    let create_info = vk::GraphicsPipelineCreateInfo::builder()
        .render_pass(render_pass)
        .subpass(subpass)
        .stages(&shader_state.stages)
        .vertex_input_state(&mesh_state.vertex_input())
        .input_assembly_state(&mesh_state.input_assembly())
        .tessellation_state(&shader_state.tessalation)
        .viewport_state(&subpass_state.viewport())
        .rasterization_state(&shader_state.rasterisation)
        .multisample_state(&subpass_state.multisample())
        .depth_stencil_state(&subpass_state.depth_stencil())
        .color_blend_state(&subpass_state.color_blend())
        .dynamic_state(&shader_state.dynamic)
        .layout(layout)
        .build();
    
    let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None) }?.0[0];

    set_object_name(instance, device, format!("{} Material Pipeline", material_name), vk::ObjectType::PIPELINE, pipeline.as_raw())?;

    Ok(pipeline)
}
