use std::ffi::CString;
use std::{fs, io};
use std::io::Read;
use std::fs::File;

use anyhow::Result;
use shaderc::{CompileOptions, Compiler, ShaderKind};

use ash::vk;
use ash::{Device, Instance};
use ash::util::read_spv;

use crate::{insert_command_label, set_object_name, Loadable, MeshData, RendererData};

pub trait Shader {
    fn stages(&self) -> Vec<PipelineShaderStage>;
    fn state(&self) -> PipelineShaderState {
        PipelineShaderState::default()
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

#[derive(Debug, Clone)]
pub struct GraphicsShader {
    pub vertex: vk::ShaderModule,
    pub fragment: vk::ShaderModule,
    pub geometry: vk::ShaderModule,
    pub tessalation_control: vk::ShaderModule,
    pub tessalation_evaluation: vk::ShaderModule,
    pub state: PipelineShaderState,
    loaded: bool,
}

impl GraphicsShader {
    pub fn builder<'a>(instance: &'a Instance, device: &'a Device, name: String) -> GraphicsShaderBuilder<'a> {
        GraphicsShaderBuilder::new(instance, device, name)
    }
}

impl Shader for GraphicsShader {
    fn stages(&self) -> Vec<PipelineShaderStage> {
        let mut stages = vec![
            PipelineShaderStage {
                shader: self.vertex,
                stage: vk::ShaderStageFlags::VERTEX,
                name: c"main".to_owned(),
            },
        ];

        if self.fragment != vk::ShaderModule::default() {
            stages.push(
                PipelineShaderStage {
                    shader: self.fragment,
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    name: c"main".to_owned(),
                },
            );
        }

        if self.geometry != vk::ShaderModule::default() {
            stages.push(
                PipelineShaderStage {
                    shader: self.geometry,
                    stage: vk::ShaderStageFlags::GEOMETRY,
                    name: c"main".to_owned(),
                },
            );
        }

        if self.tessalation_control != vk::ShaderModule::default() {
            stages.push(
                PipelineShaderStage {
                    shader: self.tessalation_control,
                    stage: vk::ShaderStageFlags::TESSELLATION_CONTROL,
                    name: c"main".to_owned(),
                },
            );
        }

        if self.tessalation_evaluation != vk::ShaderModule::default() {
            stages.push(
                PipelineShaderStage {
                    shader: self.tessalation_evaluation,
                    stage: vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                    name: c"main".to_owned(),
                },
            );
        }

        stages
    }

    fn state(&self) -> PipelineShaderState {
        let mut state = self.state.clone();
        state.stages = self.stages();
        state
    }

    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.vertex, None);
            device.destroy_shader_module(self.fragment, None);

            if self.geometry != vk::ShaderModule::default() {
                device.destroy_shader_module(self.geometry, None);
            }

            if self.tessalation_control != vk::ShaderModule::default() {
                device.destroy_shader_module(self.tessalation_control, None);
            }

            if self.tessalation_evaluation != vk::ShaderModule::default() {
                device.destroy_shader_module(self.tessalation_evaluation, None);
            }
        }
    }

    fn clone_dyn(&self) -> Box<dyn Shader> {
        todo!()
    }

    fn loaded(&self) -> bool {
        self.loaded
    }
}

pub struct GraphicsShaderBuilder<'a> {
    instance: &'a Instance,
    device: &'a Device,

    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    geometry: vk::ShaderModule,
    tessalation_control: vk::ShaderModule,
    tessalation_evaluation: vk::ShaderModule,

    state: PipelineShaderState,

    name: String,
}

impl<'a> GraphicsShaderBuilder<'a> {
    pub fn new(instance: &'a Instance, device: &'a Device, name: String) -> Self {
        GraphicsShaderBuilder {
            instance,
            device,
            vertex: vk::ShaderModule::default(),
            fragment: vk::ShaderModule::default(),
            geometry: vk::ShaderModule::default(),
            tessalation_control: vk::ShaderModule::default(),
            tessalation_evaluation: vk::ShaderModule::default(),

            state: PipelineShaderState::default(),

            name,
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
            &(self.name.clone() + " Vertex Shader"),
            vertex,
        )
        .unwrap();

        self.vertex = vertex;
        self
    }

    pub fn load_vertex(&mut self, path: &str) -> &mut Self {
        let bytecode = read_spv(&mut fs::File::open(path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(&bytecode);

        let vertex = unsafe { self.device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            &(self.name.clone() + " Vertex Shader"),
            vertex,
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
            &(self.name.clone() + " Fragment Shader"),
            fragment,
        )
        .unwrap();

        self.fragment = fragment;
        self
    }

    pub fn load_fragment(&mut self, path: &str) -> &mut Self {
        let bytecode = read_spv(&mut fs::File::open(path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(&bytecode);

        let fragment = unsafe { self.device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            &(self.name.clone() + " Fragment Shader"),
            fragment,
        )
        .unwrap();

        self.fragment = fragment;
        self
    }

    /// Compiles the geometry shader at the location from `path`
    /// If the geometry shader is not needed, this function can be skipped
    pub fn compile_geometry(&mut self, path: &str) -> &mut Self {
        let geometry =
            compile_shader_module(&self.device, path, "main", ShaderKind::Geometry)
                .unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            &(self.name.clone() + " Geometry Shader"),
            geometry,
        )
        .unwrap();

        self.geometry = geometry;
        self
    }

    pub fn load_geometry(&mut self, path: &str) -> &mut Self {
        let bytecode = read_spv(&mut fs::File::open(path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(&bytecode);

        let geometry = unsafe { self.device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            &(self.name.clone() + " Geometry Shader"),
            geometry,
        )
        .unwrap();

        self.geometry = geometry;
        self
    }

    /// Compiles the tessalation control shader at the location from `path`
    /// If the tessalation control shader is not needed, this function can be skipped
    pub fn compile_tessalation_control(&mut self, path: &str) -> &mut Self {
        let tessalation_control =
            compile_shader_module(&self.device, path, "main", ShaderKind::TessControl)
                .unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            &(self.name.clone() + " Tessalation Control Shader"),
            tessalation_control,
        )
        .unwrap();

        self.tessalation_control = tessalation_control;
        self
    }

    pub fn load_tessalation_control(&mut self, path: &str) -> &mut Self {
        let bytecode = read_spv(&mut fs::File::open(path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(&bytecode);

        let tess_ctrl = unsafe { self.device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            &(self.name.clone() + " Tessallation Control Shader"),
            tess_ctrl,
        )
        .unwrap();

        self.tessalation_control = tess_ctrl;
        self
    }

    /// Compiles the tessalation evaluation shader at the location from `path`
    /// If the tessalation evaluation shader is not needed, this function can be skipped
    pub fn compile_tessalation_evaluation(&mut self, path: &str) -> &mut Self {
        let tessalation_evaluation =
            compile_shader_module(&self.device, path, "main", ShaderKind::TessEvaluation)
                .unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            &(self.name.clone() + " Tessalation Evaluation Shader"),
            tessalation_evaluation,
        )
        .unwrap();

        self.tessalation_evaluation = tessalation_evaluation;
        self
    }

    pub fn load_tessalation_evaluation(&mut self, path: &str) -> &mut Self {
        let bytecode = read_spv(&mut fs::File::open(path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(&bytecode);

        let tess_eval = unsafe { self.device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(
            &self.instance,
            &self.device,
            &(self.name.clone() + " Tessallation Evaluation Shader"),
            tess_eval,
        )
        .unwrap();

        self.tessalation_evaluation = tess_eval;
        self
    }

    pub fn state(&mut self, state: PipelineShaderState) -> &mut Self {
        self.state = state;
        self
    }

    pub fn build(&mut self) -> GraphicsShader {
        GraphicsShader {
            vertex: self.vertex,
            fragment: self.fragment,
            geometry: self.geometry,
            tessalation_control: self.tessalation_control,
            tessalation_evaluation: self.tessalation_evaluation,

            state: self.state.clone(),

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

    let bytecode = read_spv(&mut io::Cursor::new(bytes)).unwrap();

    let info = vk::ShaderModuleCreateInfo::default()
        .code(&bytecode);

    Ok(unsafe { device.create_shader_module(&info, None) }?)
}


pub trait Material {
    fn draw(
        &self,
        instance: &Instance,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        descriptor_set: Vec<vk::DescriptorSet>,
        other_descriptors: Vec<vk::DescriptorSet>,
        mesh_data: &MeshData,
        object_mane: &str,
    );
    fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &RendererData);

    fn get_scene_descriptor_ids(&self) -> &[usize];

    fn destroy_swapchain(&self, device: &Device);
    fn destroy(&self, device: &Device);

    fn clone_dyn(&self) -> Box<dyn Material>;
    fn loaded(&self) -> bool;
}

impl Clone for Box<dyn Material> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

impl Loadable for Box<dyn Material> {
    fn is_loaded(&self) -> bool {
        self.loaded()
    }
}


#[derive(Debug, Clone)]
pub struct BasicMaterial {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,

    pub push_constant_ranges: Vec<vk::PushConstantRange>,

    pub descriptor_set_layout: Vec<vk::DescriptorSetLayout>,
    pub other_set_layouts: Vec<vk::DescriptorSetLayout>,

    pub mesh_state: PipelineMeshState,
    pub shader_state: PipelineShaderState,
    pub subpass_state: SubpassPipelineState,

    pub scene_descriptors: Vec<usize>,

    loaded: bool,
    subpass: u32,
}

impl BasicMaterial {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,
        bindings: Vec<Vec<vk::DescriptorSetLayoutBinding>>,
        push_constant_sizes: Vec<(u32, vk::ShaderStageFlags)>,
        shader_state: PipelineShaderState,
        subpass_state: SubpassPipelineState,
        mesh_state: PipelineMeshState,
        other_layouts: Vec<vk::DescriptorSetLayout>,
        scene_descriptors: Vec<usize>,
        subpass: u32,
    ) -> Self {
        let mut push_constant_ranges = vec![];
        let mut offset = 0u32;
        for (size, stage_flag) in &push_constant_sizes {
            let range = vk::PushConstantRange::default()
                .stage_flags(*stage_flag)
                .offset(offset)
                .size(*size);

            offset += size;
            push_constant_ranges.push(range);
        }

        let mut descriptor_set_layout = vec![];
        for binding in bindings {
            descriptor_set_layout.push(data.global_layout_cache.create_descriptor_set_layout(device, &binding));
        }

        let mut set_layouts = descriptor_set_layout.clone();
        set_layouts.append(&mut other_layouts.clone());

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }.unwrap();

        let pipeline = create_pipeline_from_states(instance, device, pipeline_layout, &subpass_state, &shader_state, &mesh_state, subpass, data.render_pass, "").unwrap();

        BasicMaterial {
            pipeline,
            pipeline_layout,

            push_constant_ranges,

            mesh_state: mesh_state.clone(),
            shader_state: shader_state.clone(),
            subpass_state: subpass_state.clone(),

            descriptor_set_layout,
            other_set_layouts: other_layouts.clone(),
            scene_descriptors,
            
            subpass,
            loaded: true,
        }
    }
}

impl Material for BasicMaterial {
    fn draw(
        &self,
        instance: &Instance,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        descriptor_set: Vec<vk::DescriptorSet>,
        other_descriptors: Vec<vk::DescriptorSet>,
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
                &descriptor_set,
                &[],
            );

            let mut set = descriptor_set.len() as u32;

            for descriptor in other_descriptors {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    set,
                    &[descriptor],
                    &[],
                );

                set += 1;
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
                device,
                command_buffer,
                &format!("Draw {}", name),
                [0.0, 0.5, 0.1, 1.0],
            );
            
            device.cmd_draw_indexed(command_buffer, mesh_data.index_count, 1, 0, 0, 0);
        }
    }

    fn recreate_swapchain(&mut self, instance: &Instance, device: &Device, data: &RendererData) {
        self.subpass_state.viewports[0].width = data.swapchain_extent.width as f32;
        self.subpass_state.viewports[0].height = data.swapchain_extent.height as f32;
        self.subpass_state.scissors[0].extent = data.swapchain_extent;

        let mut set_layouts = self.descriptor_set_layout.clone();

        set_layouts.append(&mut self.other_set_layouts.clone());

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&self.push_constant_ranges);

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

    fn get_scene_descriptor_ids(&self) -> &[usize] {
        &self.scene_descriptors
    }

    fn destroy_swapchain(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }

    fn destroy(&self, _device: &Device) {
        
    }

    fn clone_dyn(&self) -> Box<dyn Material> {
        Box::new(self.clone())
    }

    fn loaded(&self) -> bool {
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
        vk::PipelineViewportStateCreateInfo::default()
            .viewports(&self.viewports)
            .scissors(&self.scissors)
    }

    pub fn multisample(&self) -> vk::PipelineMultisampleStateCreateInfo {
        vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(self.rasterisation_samples)
            .sample_shading_enable(false)
    }

    pub fn depth_stencil(&self) -> vk::PipelineDepthStencilStateCreateInfo {
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(self.depth_test)
            .depth_write_enable(self.depth_write)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
    }

    pub fn color_blend(&self) -> vk::PipelineColorBlendStateCreateInfo {
        vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(self.logic_op_enable)
            .logic_op(self.logic_op)
            .attachments(&self.attachments)
            .blend_constants(self.blend_constants)
    }
}


#[derive(Default, Debug, Clone)]
pub struct PipelineShaderState {
    pub stages: Vec<PipelineShaderStage>,
    pub rasterisation: RasterisationState,
    pub tessalation: TessalationState,
    pub dynamic: DynamicState,
}

impl PipelineShaderState {
    pub fn get_rasterisation(&self) -> vk::PipelineRasterizationStateCreateInfo {
        vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(self.rasterisation.depth_clamp_enable)
            .rasterizer_discard_enable(self.rasterisation.rasterizer_discard_enable)
            .polygon_mode(self.rasterisation.polygon_mode)
            .cull_mode(self.rasterisation.cull_mode)
            .front_face(self.rasterisation.front_face)
            .depth_bias_enable(self.rasterisation.depth_bias_enable)
            .depth_bias_constant_factor(self.rasterisation.depth_bias_constant_factor)
            .depth_bias_clamp(self.rasterisation.depth_bias_clamp)
            .depth_bias_slope_factor(self.rasterisation.depth_bias_slope_factor)
            .line_width(self.rasterisation.line_width)
    }

    pub fn get_tessalation(&self) -> vk::PipelineTessellationStateCreateInfo {
        vk::PipelineTessellationStateCreateInfo::default()
            .patch_control_points(self.tessalation.patch_control_points)
    }

    pub fn get_dynamic(&self) -> vk::PipelineDynamicStateCreateInfo {
        vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&self.dynamic.dynamic_states)
    }

    pub fn get_stages(&self) -> Vec<vk::PipelineShaderStageCreateInfo> {
        self.stages.iter().map(|s| {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(s.stage)
                .module(s.shader)
                .name(&s.name)
        }).collect::<Vec<_>>()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RasterisationState {
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

impl Default for RasterisationState {
    fn default() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TessalationState {
    patch_control_points: u32,
}

#[derive(Debug, Clone, Default)]
pub struct DynamicState {
    pub dynamic_states: Vec<vk::DynamicState>,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineShaderStage {
    pub shader: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
    pub name: CString,
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
        vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&self.bindings)
            .vertex_attribute_descriptions(&self.attributes)
    }

    pub fn input_assembly(&self) -> vk::PipelineInputAssemblyStateCreateInfo {
        vk::PipelineInputAssemblyStateCreateInfo::default()
            .primitive_restart_enable(self.primitive_restart)
            .topology(self.topology)
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
    let stages = shader_state.get_stages();
    let vertex_input_state = mesh_state.vertex_input();
    let input_assembly_state = mesh_state.input_assembly();
    let tessellation_state = shader_state.get_tessalation();
    let viewport_state = subpass_state.viewport();
    let rasterization_state = shader_state.get_rasterisation();
    let multisample_state = subpass_state.multisample();
    let depth_stencil_state = subpass_state.depth_stencil();
    let color_blend_state = subpass_state.color_blend();
    let dynamic_state = shader_state.get_dynamic();

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .render_pass(render_pass)
        .subpass(subpass)
        .stages(&stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .tessellation_state(&tessellation_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(layout);
    
    let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None) }
        .map_err(|(_, err)| anyhow::anyhow!("Failed to create graphics pipeline: {:?}", err))?[0];

    set_object_name(
        instance,
        device,
        &format!("{} Material Pipeline", material_name),
        pipeline
    )?;

    Ok(pipeline)
}
