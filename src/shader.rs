use std::{ffi::CString, fs, io::Cursor, path::Path};

use ash::{Device, Instance, util::read_spv, vk};
use slang_rs::{convert::get_shader_stage, enums::CompileTarget, interfaces::{AsComponentType, GlobalSession}, structs::{SessionDesc, TargetDesc}};

use crate::{Vertex, set_object_name};

#[derive(Debug, Clone)]
pub struct GraphicsProgram<'a> {
    name: &'a str,
    vertex_path: &'a str,
    vertex_module: Option<vk::PipelineShaderStageCreateInfo<'a>>,
    tess_ctrl_path: Option<&'a str>,
    tess_ctrl_module: Option<vk::PipelineShaderStageCreateInfo<'a>>,
    tess_eval_path: Option<&'a str>,
    tess_eval_module: Option<vk::PipelineShaderStageCreateInfo<'a>>,
    geometry_path: Option<&'a str>,
    geometry_module: Option<vk::PipelineShaderStageCreateInfo<'a>>,
    fragment_path: Option<&'a str>,
    fragment_module: Option<vk::PipelineShaderStageCreateInfo<'a>>,
}

impl<'a> GraphicsProgram<'a> {
    pub fn new(name: &'a str, vertex_shader_path: &'a str) -> Self {
        GraphicsProgram {
            name,
            vertex_path: vertex_shader_path,
            vertex_module: None,
            tess_ctrl_path: None,
            tess_ctrl_module: None,
            tess_eval_path: None,
            tess_eval_module: None,
            geometry_path: None,
            geometry_module: None,
            fragment_path: None,
            fragment_module: None,
        }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn vertex_path(&self) -> &'a str {
        self.vertex_path
    }

    pub fn tess_ctrl_path(&self) -> &Option<&'a str> {
        &self.tess_ctrl_path
    }

    pub fn tess_eval_path(&self) -> &Option<&'a str> {
        &self.tess_eval_path
    }

    pub fn geometry_path(&self) -> &Option<&'a str> {
        &self.geometry_path
    }

    pub fn fragment_path(&self) -> &Option<&'a str> {
        &self.fragment_path
    }

    pub fn vertex_module(&self) -> Option<vk::PipelineShaderStageCreateInfo<'a>> {
        self.vertex_module
    }

    pub fn tess_ctrl_module(&self) -> Option<vk::PipelineShaderStageCreateInfo<'a>> {
        self.tess_ctrl_module
    }

    pub fn tess_eval_module(&self) -> Option<vk::PipelineShaderStageCreateInfo<'a>> {
        self.tess_eval_module
    }

    pub fn geometry_module(&self) -> Option<vk::PipelineShaderStageCreateInfo<'a>> {
        self.geometry_module
    }

    pub fn fragment_module(&self) -> Option<vk::PipelineShaderStageCreateInfo<'a>> {
        self.fragment_module
    }
}

impl<'a> GraphicsProgram<'a> {
    pub fn set_tess_ctrl_path(&mut self, tess_ctrl_path: &'a str) {
        self.tess_ctrl_path = Some(tess_ctrl_path)
    }

    pub fn set_tess_eval_path(&mut self, tess_eval_path: &'a str) {
        self.tess_eval_path = Some(tess_eval_path)
    }

    pub fn set_geometry_path(&mut self, geometry_path: &'a str) {
        self.geometry_path = Some(geometry_path)
    }

    pub fn set_fragment_path(&mut self, fragment_path: &'a str) {
        self.fragment_path = Some(fragment_path)
    }

    pub fn load_shader_modules_spirv(&mut self, instance: &Instance, device: &Device) {
        let bytecode = read_spv(&mut fs::File::open(self.vertex_path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(&bytecode);

        let vertex = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(instance, device, &format!("{} Vertex Shader", self.name), vertex).unwrap();

        let vertex_module = vk::PipelineShaderStageCreateInfo::default()
            .module(vertex)
            .name(c"main")
            .stage(vk::ShaderStageFlags::VERTEX);

        self.vertex_module = Some(vertex_module);

        if let Some(tess_ctrl_path) = self.tess_ctrl_path {
            let bytecode = read_spv(&mut fs::File::open(tess_ctrl_path).unwrap()).unwrap();
            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(&bytecode);

            let tess_ctrl = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

            set_object_name(instance, device, &format!("{} Tessalation Control Shader", self.name), tess_ctrl).unwrap();

            let tess_ctrl_module = vk::PipelineShaderStageCreateInfo::default()
                .module(tess_ctrl)
                .name(c"main")
                .stage(vk::ShaderStageFlags::TESSELLATION_CONTROL);

            self.tess_ctrl_module = Some(tess_ctrl_module);
        }

        if let Some(tess_eval_path) = self.tess_eval_path {
            let bytecode = read_spv(&mut fs::File::open(tess_eval_path).unwrap()).unwrap();
            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(&bytecode);

            let tess_eval = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

            set_object_name(instance, device, &format!("{} Tessalation Evaluation Shader", self.name), tess_eval).unwrap();

            let tess_eval_module = vk::PipelineShaderStageCreateInfo::default()
                .module(tess_eval)
                .name(c"main")
                .stage(vk::ShaderStageFlags::TESSELLATION_EVALUATION);

            self.tess_eval_module = Some(tess_eval_module);
        }

        if let Some(geometry_path) = self.geometry_path {
            let bytecode = read_spv(&mut fs::File::open(geometry_path).unwrap()).unwrap();
            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(&bytecode);

            let geometry = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

            set_object_name(instance, device, &format!("{} Geometry Shader", self.name), geometry).unwrap();

            let geometry_module = vk::PipelineShaderStageCreateInfo::default()
                .module(geometry)
                .name(c"main")
                .stage(vk::ShaderStageFlags::GEOMETRY);

            self.geometry_module = Some(geometry_module);
        }

        if let Some(fragment_path) = self.fragment_path {
            let bytecode = read_spv(&mut fs::File::open(fragment_path).unwrap()).unwrap();
            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(&bytecode);

            let fragment = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

            set_object_name(instance, device, &format!("{} Fragment Shader", self.name), fragment).unwrap();

            let fragment_module = vk::PipelineShaderStageCreateInfo::default()
                .module(fragment)
                .name(c"main")
                .stage(vk::ShaderStageFlags::FRAGMENT);

            self.fragment_module = Some(fragment_module);
        }
    }

    pub fn get_stages(&self) -> Vec<vk::PipelineShaderStageCreateInfo<'a>> {
        let mut stages = vec![];

        if let Some(vertex_module) = self.vertex_module {
            stages.push(vertex_module);
        }

        if let Some(tess_ctrl_module) = self.tess_ctrl_module {
            stages.push(tess_ctrl_module);
        }

        if let Some(tess_eval_module) = self.tess_eval_module {
            stages.push(tess_eval_module);
        }

        if let Some(geometry_module) = self.geometry_module {
            stages.push(geometry_module);
        }

        if let Some(fragment_module) = self.fragment_module {
            stages.push(fragment_module);
        }

        stages
    }

    pub fn destroy(&mut self, device: &Device) {
        if let Some(vertex_module) = self.vertex_module {
            unsafe { device.destroy_shader_module(vertex_module.module, None) };
        }
        self.vertex_module = None;

        if let Some(tess_ctrl_module) = self.tess_ctrl_module {
            unsafe { device.destroy_shader_module(tess_ctrl_module.module, None) };
        }
        self.tess_ctrl_module = None;

        if let Some(tess_eval_module) = self.tess_eval_module {
            unsafe { device.destroy_shader_module(tess_eval_module.module, None) };
        }
        self.tess_eval_module = None;

        if let Some(geometry_module) = self.geometry_module {
            unsafe { device.destroy_shader_module(geometry_module.module, None) };
        }
        self.geometry_module = None;

        if let Some(fragment_module) = self.fragment_module {
            unsafe { device.destroy_shader_module(fragment_module.module, None) };
        }
        self.fragment_module = None;
    }
}

pub fn create_basic_graphics_pipeline<'a, V: Vertex>(
    instance: &Instance,
    device: &Device,
    program: &GraphicsProgram,
    rasterization_data: RasterizationData,
    multisample_data: MultisampleData,
    depth_stencil_data: DepthStencilData,
    patch_control_points: u32,
    attachment_blends: &[vk::PipelineColorBlendAttachmentState],
    color_attachment_formats: &[vk::Format],
    depth_attachment_format: vk::Format,
    stencil_attachment_format: vk::Format,
    push_constant_ranges: &[vk::PushConstantRange],
    set_layouts: &[vk::DescriptorSetLayout],
    topology: vk::PrimitiveTopology,
) -> (vk::Pipeline, vk::PipelineLayout) {
    let attribute_descriptions = V::attribute_descriptions();
    let binding_descriptions = V::binding_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&attribute_descriptions)
        .vertex_binding_descriptions(&binding_descriptions);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .primitive_restart_enable(false)
        .topology(topology);

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let tessellation_state = vk::PipelineTessellationStateCreateInfo::default()
        .patch_control_points(patch_control_points);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissor_count(1)
        .viewport_count(1);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_bias_enable(false)
        .depth_clamp_enable(false)
        .cull_mode(rasterization_data.cull_mode)
        .front_face(rasterization_data.front_face)
        .line_width(rasterization_data.line_width)
        .polygon_mode(rasterization_data.polygon_mode)
        .rasterizer_discard_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(multisample_data.samples)
        .sample_shading_enable(multisample_data.sample_shading_enable);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_stencil_data.depth_test_enable)
        .depth_write_enable(depth_stencil_data.depth_write_enable)
        .stencil_test_enable(depth_stencil_data.stencil_test_enable)
        .depth_compare_op(depth_stencil_data.compare_op);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(attachment_blends);

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&push_constant_ranges)
        .set_layouts(&set_layouts);

    let layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }.unwrap();

    set_object_name(instance, device, &format!("{} Pipeline Layout", program.name()), layout).unwrap();

    let mut pipeline_info2 = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(color_attachment_formats)
        .depth_attachment_format(depth_attachment_format)
        .stencil_attachment_format(stencil_attachment_format);

    let stages = program.get_stages();
    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
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
        .layout(layout)
        .push_next(&mut pipeline_info2);

    let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None) }.unwrap()[0];

    set_object_name(instance, device, &format!("{} Pipeline", program.name()), pipeline).unwrap();

    (pipeline, layout)
}

pub struct RasterizationData {
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub polygon_mode: vk::PolygonMode,
    pub line_width: f32,
}

pub struct MultisampleData {
    pub samples: vk::SampleCountFlags,
    pub sample_shading_enable: bool,
}

pub struct DepthStencilData {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub stencil_test_enable: bool,
    pub compare_op: vk::CompareOp,
}

pub struct ComputeProgram<'a> {
    name: &'a str,
    compute_path: &'a str,
    compute_module: Option<vk::PipelineShaderStageCreateInfo<'a>>,
}

impl<'a> ComputeProgram<'a> {
    pub fn new(name: &'a str, path: &'a str) -> Self {
        ComputeProgram {
            name,
            compute_path: path,
            compute_module: None,
        }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn path(&self) -> &str {
        self.compute_path
    }

    pub fn compute_module(&self) -> Option<vk::PipelineShaderStageCreateInfo<'a>> {
        self.compute_module
    }

    pub fn load_shader_module_spirv(&mut self, instance: &Instance, device: &Device) {
        let bytecode = read_spv(&mut fs::File::open(self.compute_path).unwrap()).unwrap();
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(&bytecode);

        let compute = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

        set_object_name(instance, device, &format!("{} Compute Shader", self.name), compute).unwrap();

        let compute_module = vk::PipelineShaderStageCreateInfo::default()
            .module(compute)
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE);

        self.compute_module = Some(compute_module);
    }
}

pub fn create_compute_pipeline(
    instance: &Instance,
    device: &Device,
    program: &ComputeProgram,
    push_constant_ranges: &[vk::PushConstantRange],
    set_layouts: &[vk::DescriptorSetLayout],
) -> (vk::Pipeline, vk::PipelineLayout) {

    let layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);

    let layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }.unwrap();
    set_object_name(instance, device, &format!("{} Pipeline Layout", program.name()), layout).unwrap();

    let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
        .stage(program.compute_module.unwrap())
        .layout(layout);

    let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None).unwrap()[0] };
    set_object_name(instance, device, &format!("{} Pipeline", program.name()), pipeline).unwrap();

    (pipeline, layout)
}


#[derive(Debug, Clone)]
pub struct SlangShader<'a> {
    name: &'a str,
    path: &'a Path,
    program: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
}

impl<'a> SlangShader<'a> {
    pub fn new(name: &'a str, path: &'a Path) -> Self {
        SlangShader {
            name,
            path,
            program: vec![],
        }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn path(&self) -> &Path {
        self.path
    }

    pub fn program(&self) -> &[vk::PipelineShaderStageCreateInfo<'a>] {
        &self.program
    }

    pub fn load_and_compile(&mut self, device: &Device) {
        let global_session = GlobalSession::new().expect("failed to create a global session");

        let spriv_profile = global_session.find_profile("spirv_1_5");

        let target = TargetDesc::default()
            .format(CompileTarget::SPIRV)
            .profile(spriv_profile);

        let targets = [target.into()];
        let cstring = CString::new(self.path.parent().unwrap().to_str().unwrap()).unwrap();
        let paths = [cstring.as_ptr()];
        let session_desc = SessionDesc::default()
            .targets(&targets)
            .search_paths(&paths)
            .compiler_option_entries(&[]);
            // .default_matrix_layout_mode(slang_rs::enums::MatrixLayoutMode::ColumnMajor);

        let session = global_session.create_session(&session_desc.into()).expect("could not create session");

        let module = session.load_module(self.path.file_name().unwrap().to_str().unwrap()).unwrap();

        let entry_point_count = module.get_defined_entry_point_count();
        let entry_points = (0..entry_point_count).into_iter().map(|i| module.get_defined_entry_point(i).unwrap()).collect::<Vec<_>>();

        let mut components = vec![Box::new(module) as _];

        entry_points.into_iter().for_each(|e| components.push(Box::new(e) as _));

        let program = session.create_composite_component_type(&components).unwrap();

        let linked_module = program.link().unwrap();

        let program_layout = linked_module.get_layout(0).unwrap();
        
        let program = (0..entry_point_count).into_iter().map(|i| {
            let ep_layout = program_layout.get_entry_point_by_index(i);
            let stage = get_shader_stage(&ep_layout.get_stage()).unwrap();
            
            let blob = linked_module.get_entry_point_code(i, 0).unwrap();
            let code = blob.as_bytes();

            let mut cursor = Cursor::new(code);

            let bytecode = read_spv(&mut cursor).unwrap();

            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(&bytecode);

            let module = unsafe { device.create_shader_module(&create_info, None) }.unwrap();


            vk::PipelineShaderStageCreateInfo::default()
                .module(module)
                .name(c"main")
                .stage(stage)
        }).collect::<Vec<_>>();

        self.program = program;
    }

    pub fn destroy(&mut self, device: &Device) {
        self.program.iter().for_each(|s| unsafe { device.destroy_shader_module(s.module, None) });
        self.program.clear();
    }
}

pub fn create_basic_slang_graphics_pipeline<'a, V: Vertex>(
    instance: &Instance,
    device: &Device,
    program: &SlangShader,
    rasterization_data: RasterizationData,
    multisample_data: MultisampleData,
    depth_stencil_data: DepthStencilData,
    patch_control_points: u32,
    attachment_blends: &[vk::PipelineColorBlendAttachmentState],
    color_attachment_formats: &[vk::Format],
    depth_attachment_format: vk::Format,
    stencil_attachment_format: vk::Format,
    push_constant_ranges: &[vk::PushConstantRange],
    set_layouts: &[vk::DescriptorSetLayout],
    topology: vk::PrimitiveTopology,
) -> (vk::Pipeline, vk::PipelineLayout) {
    let attribute_descriptions = V::attribute_descriptions();
    let binding_descriptions = V::binding_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&attribute_descriptions)
        .vertex_binding_descriptions(&binding_descriptions);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .primitive_restart_enable(false)
        .topology(topology);

    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let tessellation_state = vk::PipelineTessellationStateCreateInfo::default()
        .patch_control_points(patch_control_points);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissor_count(1)
        .viewport_count(1);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_bias_enable(false)
        .depth_clamp_enable(false)
        .cull_mode(rasterization_data.cull_mode)
        .front_face(rasterization_data.front_face)
        .line_width(rasterization_data.line_width)
        .polygon_mode(rasterization_data.polygon_mode)
        .rasterizer_discard_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(multisample_data.samples)
        .sample_shading_enable(multisample_data.sample_shading_enable);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_stencil_data.depth_test_enable)
        .depth_write_enable(depth_stencil_data.depth_write_enable)
        .stencil_test_enable(depth_stencil_data.stencil_test_enable)
        .depth_compare_op(depth_stencil_data.compare_op);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(attachment_blends);

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&push_constant_ranges)
        .set_layouts(&set_layouts);

    let layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }.unwrap();

    set_object_name(instance, device, &format!("{} Pipeline Layout", program.name()), layout).unwrap();

    let mut pipeline_info2 = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(color_attachment_formats)
        .depth_attachment_format(depth_attachment_format)
        .stencil_attachment_format(stencil_attachment_format);

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(program.program())
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .tessellation_state(&tessellation_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .push_next(&mut pipeline_info2);

    let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None) }.unwrap()[0];

    set_object_name(instance, device, &format!("{} Pipeline", program.name()), pipeline).unwrap();

    (pipeline, layout)
}
