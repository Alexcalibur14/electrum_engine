#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use anyhow::Result;
use glam::{vec2, vec3, Mat4, Vec2, Vec3};
use vulkanalia::prelude::v1_2::*;

use crate::buffer::{copy_buffer, create_buffer, BufferWrapper};
use crate::shader::{Material, PipelineMeshSettings, VFShader};
use crate::{Image, PointLight, RenderStats, RendererData};

pub trait Renderable {
    unsafe fn draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        camera_data: vk::DescriptorSet,
    );
    fn update(
        &mut self,
        device: &Device,
        data: &RendererData,
        stats: &RenderStats,
        index: usize,
        view_proj: Mat4,
        id: usize,
    );
    fn destroy_swapchain(&self, device: &Device);
    fn recreate_swapchain(&mut self, device: &Device, data: &RendererData);
    fn destroy(&self, device: &Device);
    fn clone_dyn(&self) -> Box<dyn Renderable>;
}

impl Clone for Box<dyn Renderable> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

pub trait Vertex {
    fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription>;
    fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>;
}

#[derive(Debug, Clone, Default)]
pub struct Descriptors {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,

    pub bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl Descriptors {
    pub fn new(
        device: &Device,
        data: &RendererData,
        bindings: Vec<vk::DescriptorSetLayoutBinding>,
    ) -> Self {
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&info, None) }.unwrap();

        let descriptor_pool =
            unsafe { Descriptors::create_descriptor_pool(device, data, &bindings) }.unwrap();

        Descriptors {
            descriptor_set_layout,
            descriptor_pool,

            bindings,
        }
    }

    pub unsafe fn create_descriptor_pool(
        device: &Device,
        data: &RendererData,
        bindings: &Vec<vk::DescriptorSetLayoutBinding>,
    ) -> Result<vk::DescriptorPool> {
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

    pub fn destroy_swapchain(&self, device: &Device) {
        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };
    }

    pub fn recreate_swapchain(&mut self, device: &Device, data: &RendererData) {
        self.descriptor_pool =
            unsafe { Self::create_descriptor_pool(device, data, &self.bindings) }.unwrap();
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}

#[derive(Clone)]
pub struct ObjectPrototype {
    vertices: Vec<PCTVertex>,
    indices: Vec<u32>,

    vertex_buffer: BufferWrapper,
    index_buffer: BufferWrapper,

    material: Material,

    ubo: UniformBufferObject,
    ubo_buffers: Vec<BufferWrapper>,

    image: (Image, vk::Sampler),

    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Renderable for ObjectPrototype {
    unsafe fn draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        camera_data: vk::DescriptorSet,
    ) {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.material.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.material.pipeline_layout,
            0,
            &[self.descriptor_sets[image_index]],
            &[],
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.material.pipeline_layout,
            1,
            &[camera_data],
            &[],
        );
        device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer.buffer], &[0]);
        device.cmd_bind_index_buffer(
            command_buffer,
            self.index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );
        device.cmd_draw_indexed(command_buffer, self.indices.len() as u32, 1, 0, 0, 0);
    }

    fn destroy_swapchain(&self, device: &Device) {
        self.material.destroy_swapchain(device);
    }

    fn recreate_swapchain(&mut self, device: &Device, data: &RendererData) {
        self.material.recreate_swapchain(device, data);

        let layouts =
            vec![self.material.descriptor.descriptor_set_layout; data.swapchain_images.len()];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.material.descriptor.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info) }.unwrap();

        for (set_index, descriptor_set) in descriptor_sets
            .iter()
            .enumerate()
            .take(data.swapchain_images.len())
        {
            let info = vk::DescriptorBufferInfo::builder()
                .buffer(self.ubo_buffers[set_index].buffer)
                .offset(0)
                .range(size_of::<ModelData>() as u64)
                .build();

            let buffer_info = &[info];
            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(*descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info)
                .build();

            let info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.image.0.view)
                .sampler(self.image.1)
                .build();

            let image_info = &[info];
            let texture_write = vk::WriteDescriptorSet::builder()
                .dst_set(*descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info)
                .build();

            let info = vk::DescriptorBufferInfo::builder()
                .buffer(data.point_lights[0][set_index].buffer)
                .offset(0)
                .range(size_of::<PointLight>() as u64)
                .build();

            let buffer_info = &[info];
            let light_write = vk::WriteDescriptorSet::builder()
                .dst_set(*descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info)
                .build();

            unsafe {
                device.update_descriptor_sets(
                    &[ubo_write, texture_write, light_write],
                    &[] as &[vk::CopyDescriptorSet],
                )
            };
        }
        self.descriptor_sets = descriptor_sets;
    }

    fn destroy(&self, device: &Device) {
        self.material.destroy(device);
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
        self.ubo_buffers.iter().for_each(|b| b.destroy(device));
        unsafe {
            self.image.0.destroy(device);
            device.destroy_sampler(self.image.1, None);
        }
    }

    fn clone_dyn(&self) -> Box<dyn Renderable> {
        Box::new(self.clone())
    }

    fn update(
        &mut self,
        device: &Device,
        _data: &RendererData,
        stats: &RenderStats,
        index: usize,
        view_proj: Mat4,
        _id: usize,
    ) {
        self.ubo.model = Mat4::from_translation(vec3(0.0, 0.0, 0.0))
            * Mat4::from_axis_angle(Vec3::Y, stats.start.elapsed().as_secs_f32() / 2.0);

        let mvp = view_proj * self.ubo.model;

        let normal = self.ubo.model.inverse().transpose();

        let model_data = ModelData {
            model: self.ubo.model,
            mvp,
            normal,
        };

        let mem = unsafe {
            device.map_memory(
                self.ubo_buffers[index].memory,
                0,
                size_of::<ModelData>() as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .unwrap();

        unsafe { memcpy(&model_data, mem.cast(), 1) };

        unsafe { device.unmap_memory(self.ubo_buffers[index].memory) };
    }
}

impl ObjectPrototype {
    pub fn load(
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        path: &str,
        shader: VFShader,
        model: Mat4,
        view: Mat4,
        proj: Mat4,
        image: (Image, vk::Sampler),
        light: Vec<BufferWrapper>,
        other_layouts: Vec<vk::DescriptorSetLayout>,
    ) -> Self {
        let (vertices, indices) = load_model_temp(path).unwrap();

        let mesh_settings = PipelineMeshSettings {
            binding_descriptions: PCTVertex::binding_descriptions(),
            attribute_descriptions: PCTVertex::attribute_descriptions(),
            front_face: vk::FrontFace::CLOCKWISE,
            ..Default::default()
        };

        let bindings = vec![
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];

        let material = Material::new(
            device,
            data,
            bindings,
            vec![],
            shader,
            mesh_settings,
            other_layouts,
        );

        let ubo = UniformBufferObject { model, view, proj };

        let mvp = proj * view * model;

        let normal = ubo.model.inverse().transpose();

        let model_data = ModelData {
            model: ubo.model,
            mvp,
            normal,
        };

        let mut ubo_buffers = vec![];

        for i in 0..data.swapchain_images.len() {
            let buffer = unsafe {
                create_buffer(
                    instance,
                    device,
                    data,
                    size_of::<ModelData>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    Some(format!("Object UBO {}", i)),
                )
            }
            .unwrap();

            let ubo_mem = unsafe {
                device.map_memory(
                    buffer.memory,
                    0,
                    size_of::<ModelData>() as u64,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .unwrap();

            unsafe { memcpy(&model_data, ubo_mem.cast(), 1) };

            unsafe { device.unmap_memory(buffer.memory) };

            ubo_buffers.push(buffer);
        }

        let layouts = vec![material.descriptor.descriptor_set_layout; data.swapchain_images.len()];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(material.descriptor.descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info) }.unwrap();

        for i in 0..data.swapchain_images.len() {
            let info = vk::DescriptorBufferInfo::builder()
                .buffer(ubo_buffers[i].buffer)
                .offset(0)
                .range(size_of::<ModelData>() as u64)
                .build();

            let buffer_info = &[info];
            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info)
                .build();

            let info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image.0.view)
                .sampler(image.1)
                .build();

            let image_info = &[info];
            let texture_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info)
                .build();

            let info = vk::DescriptorBufferInfo::builder()
                .buffer(light[i].buffer)
                .offset(0)
                .range(size_of::<PointLight>() as u64)
                .build();

            let buffer_info = &[info];
            let light_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info)
                .build();

            unsafe {
                device.update_descriptor_sets(
                    &[ubo_write, texture_write, light_write],
                    &[] as &[vk::CopyDescriptorSet],
                )
            };
        }

        ObjectPrototype {
            vertices,
            indices,

            vertex_buffer: BufferWrapper::default(),
            index_buffer: BufferWrapper::default(),

            material,

            ubo,
            ubo_buffers,

            descriptor_sets,
            image,
        }
    }

    pub unsafe fn generate_vertex_buffer(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &RendererData,
    ) {
        let size = (size_of::<PCTVertex>() * self.vertices.len()) as u64;

        let staging_buffer = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            None,
        )
        .unwrap();

        let memory = device
            .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())
            .unwrap();

        memcpy(self.vertices.as_ptr(), memory.cast(), self.vertices.len());

        device.unmap_memory(staging_buffer.memory);

        let vertex_buffer = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            Some("Vertex Buffer".to_string()),
        )
        .unwrap();

        copy_buffer(
            device,
            staging_buffer.buffer,
            vertex_buffer.buffer,
            size,
            data,
        )
        .unwrap();

        device.destroy_buffer(staging_buffer.buffer, None);
        device.free_memory(staging_buffer.memory, None);

        self.vertex_buffer = vertex_buffer;
    }

    pub unsafe fn generate_index_buffer(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &RendererData,
    ) {
        let size = (size_of::<u32>() * self.indices.len()) as u64;

        let staging_buffer = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            None,
        )
        .unwrap();

        let memory = device
            .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())
            .unwrap();

        memcpy(self.indices.as_ptr(), memory.cast(), self.indices.len());

        device.unmap_memory(staging_buffer.memory);

        let index_buffer = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            Some("Index Buffer".to_string()),
        )
        .unwrap();

        copy_buffer(
            device,
            staging_buffer.buffer,
            index_buffer.buffer,
            size,
            data,
        )
        .unwrap();

        device.destroy_buffer(staging_buffer.buffer, None);
        device.free_memory(staging_buffer.memory, None);

        self.index_buffer = index_buffer;
    }
}

fn load_model_temp(path: &str) -> Result<(Vec<PCTVertex>, Vec<u32>)> {
    let mut reader = BufReader::new(File::open(path)?);

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: false,
            ..Default::default()
        },
        |_| Ok(Default::default()),
    )?;

    let mut unique_vertices = HashMap::new();

    let mut vertices = vec![];
    let mut indices = vec![];

    for model in &models {
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            let vertex = PCTVertex {
                pos: vec3(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                tex_coord: vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
                normal: vec3(
                    model.mesh.normals[pos_offset],
                    model.mesh.normals[pos_offset + 1],
                    model.mesh.normals[pos_offset + 2],
                ),
            };

            if let Some(index) = unique_vertices.get(&vertex) {
                indices.push(*index as u32);
            } else {
                let index = vertices.len();
                unique_vertices.insert(vertex, index);
                vertices.push(vertex);
                indices.push(index as u32);
            }
        }
    }

    Ok((vertices, indices))
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PCTVertex {
    pub pos: Vec3,
    pub tex_coord: Vec2,
    pub normal: Vec3,
}

impl Vertex for PCTVertex {
    fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<PCTVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(size_of::<Vec3>() as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec3>() as u32 + size_of::<Vec2>() as u32)
                .build(),
        ]
    }
}

impl PartialEq for PCTVertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.tex_coord == other.tex_coord && self.normal == other.normal
    }
}

impl Eq for PCTVertex {}

impl Hash for PCTVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
        self.normal[0].to_bits().hash(state);
        self.normal[1].to_bits().hash(state);
        self.normal[2].to_bits().hash(state);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ModelData {
    model: Mat4,
    mvp: Mat4,
    normal: Mat4,
}
