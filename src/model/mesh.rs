#![allow(dead_code)]
use std::{collections::HashMap, fs::File, hash::Hash, io::BufReader, mem::size_of};

use anyhow::Result;
use glam::{vec2, vec3, Mat4, Vec3};
use vulkanalia::prelude::v1_2::*;

use crate::{
    buffer::{create_buffer, BufferWrapper}, create_and_stage_buffer, vertices::PCTVertex, DescriptorBuilder, RenderStats, Renderable, RendererData
};

use super::{ModelData, ModelMVP};

#[derive(Clone)]
pub struct Quad {
    name: String,

    points: [Vec3; 4],
    normal: Vec3,

    vertices: Vec<PCTVertex>,
    indices: Vec<u32>,

    vertex_buffer: BufferWrapper,
    index_buffer: BufferWrapper,

    image: (u64, u64),

    ubo: ModelMVP,
    ubo_buffers: Vec<BufferWrapper>,

    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Quad {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,

        points: [Vec3; 4],
        normal: Vec3,

        image: (u64, u64),

        model: Mat4,
        view: Mat4,
        proj: Mat4,

        name: String,
    ) -> Self {
        let ubo = ModelMVP { model, view, proj };

        let model_data = ubo.get_data(&(proj * view));

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
                    Some(format!("{} UBO {}", name, i)),
                )
            }
            .unwrap();

            buffer.copy_data_into_buffer(device, &model_data);

            ubo_buffers.push(buffer);
        }
        
        let sampler = data.samplers.get(&image.1).unwrap();
        let texture = data.textures.get(&image.0).unwrap();

        let mut descriptor_sets = vec![];

        for i in 0..data.swapchain_images.len() {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(ubo_buffers[i].buffer)
                .offset(0)
                .range(size_of::<ModelData>() as u64)
                .build();
            
            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture.view)
                .sampler(*sampler)
                .build();

            let (set, _) = DescriptorBuilder::new()
                .bind_buffer(0, 1, &[buffer_info], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
                .bind_image(1, 1, &[image_info], vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
                .build(device, &mut data.global_layout_cache, &mut data.global_descriptor_pools).unwrap();

            descriptor_sets.push(set);
        }

        Quad {
            name,
            vertices: vec![],
            indices: vec![],
            vertex_buffer: BufferWrapper::default(),
            index_buffer: BufferWrapper::default(),
            image,
            ubo,
            ubo_buffers,
            descriptor_sets,
            points,
            normal,
        }
    }

    pub fn generate(&mut self, instance: &Instance, device: &Device, data: &RendererData) {
        let mut vertices = vec![];

        let uvs = [
            vec2(0.0, 0.0),
            vec2(1.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 1.0),
        ];

        for (i, point) in self.points.iter().enumerate() {
            vertices.push(PCTVertex {
                pos: *point,
                tex_coord: uvs[i],
                normal: self.normal,
            })
        }

        let vertex_size = (size_of::<PCTVertex>() * vertices.len()) as u64;

        self.vertex_buffer = create_and_stage_buffer(
            instance,
            device,
            data,
            vertex_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            Some(format!("{} Vertex Buffer", self.name)),
            &vertices,
        )
        .unwrap();

        let indices: Vec<u32> = vec![0, 3, 1, 0, 2, 3];
        let index_size = (size_of::<u32>() * indices.len()) as u64;

        self.index_buffer = create_and_stage_buffer(
            instance,
            device,
            data,
            index_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            Some(format!("{} Index Buffer", self.name)),
            &indices,
        )
        .unwrap();
    }
}

impl Renderable for Quad {
    fn update(
        &mut self,
        device: &Device,
        _data: &RendererData,
        _stats: &RenderStats,
        index: usize,
        view_proj: Mat4,
    ) {
        let model_data = self.ubo.get_data(&view_proj);

        self.ubo_buffers[index].copy_data_into_buffer(device, &model_data);
    }

    fn destroy_swapchain(&self, _device: &Device) {
    }

    fn recreate_swapchain(&mut self, _device: &Device, _data: &mut RendererData) {
    }

    fn destroy(&mut self, device: &Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
        self.ubo_buffers.iter().for_each(|b| b.destroy(device));
    }

    fn clone_dyn(&self) -> Box<dyn Renderable> {
        Box::new(self.clone())
    }
    
    fn descriptor_set(&self, image_index: usize) -> vk::DescriptorSet {
        self.descriptor_sets[image_index]
    }
    
    fn vertex_buffer(&self) -> BufferWrapper {
        self.vertex_buffer
    }
    
    fn index_buffer(&self) -> BufferWrapper {
        self.index_buffer
    }

    fn index_len(&self) -> u32 {
        6
    }
    
    fn name(&self) -> String {
        self.name.clone()
    }
}

impl Hash for Quad {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.vertices.hash(state);
        self.indices.hash(state);
        self.image.hash(state);
    }
}

#[derive(Clone)]
pub struct ObjectPrototype {
    name: String,

    vertices: Vec<PCTVertex>,
    indices: Vec<u32>,

    vertex_buffer: BufferWrapper,
    index_buffer: BufferWrapper,

    ubo: ModelMVP,
    ubo_buffers: Vec<BufferWrapper>,

    image: (u64, u64),

    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl ObjectPrototype {
    pub fn load(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,
        path: &str,
        model: Mat4,
        view: Mat4,
        proj: Mat4,
        image: (u64, u64),
        name: String,
    ) -> Self {
        let (vertices, indices) = load_model_temp(path).unwrap();

        let ubo = ModelMVP { model, view, proj };

        let model_data = ubo.get_data(&(proj * view));

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
                    Some(format!("{} UBO {}", name, i)),
                )
            }
            .unwrap();

            buffer.copy_data_into_buffer(device, &model_data);

            ubo_buffers.push(buffer);
        }

        let sampler = data.samplers.get(&image.1).unwrap();
        let texture = data.textures.get(&image.0).unwrap();

        let mut descriptor_sets = vec![];

        for i in 0..data.swapchain_images.len() {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(ubo_buffers[i].buffer)
                .offset(0)
                .range(size_of::<ModelData>() as u64)
                .build();
            
            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture.view)
                .sampler(*sampler)
                .build();

            let (set, _) = DescriptorBuilder::new()
                .bind_buffer(0, 1, &[buffer_info], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
                .bind_image(1, 1, &[image_info], vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
                .build(device, &mut data.global_layout_cache, &mut data.global_descriptor_pools).unwrap();

            descriptor_sets.push(set);
        }

        ObjectPrototype {
            name,

            vertices,
            indices,

            vertex_buffer: BufferWrapper::default(),
            index_buffer: BufferWrapper::default(),

            ubo,
            ubo_buffers,

            descriptor_sets,
            image,
        }
    }

    pub fn generate_vertex_buffer(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &RendererData,
    ) {
        let size = (size_of::<PCTVertex>() * self.vertices.len()) as u64;

        self.vertex_buffer = create_and_stage_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            Some(format!("{} Vertex Buffer", self.name)),
            &self.vertices,
        )
        .unwrap();
    }

    pub fn generate_index_buffer(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &RendererData,
    ) {
        let size = (size_of::<u32>() * self.indices.len()) as u64;

        self.index_buffer = create_and_stage_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            Some(format!("{} Index Buffer", self.name)),
            &self.indices,
        )
        .unwrap();
    }
}

impl Renderable for ObjectPrototype {
    fn destroy_swapchain(&self, _device: &Device) {
    }

    fn recreate_swapchain(&mut self, _device: &Device, _data: &mut RendererData) {
    }

    fn destroy(&mut self, device: &Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
        self.ubo_buffers.iter().for_each(|b| b.destroy(device));
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
    ) {
        self.ubo.model = Mat4::from_translation(vec3(0.0, 0.0, 0.0))
            * Mat4::from_axis_angle(Vec3::Y, stats.start.elapsed().as_secs_f32() / 2.0);

        let model_data = self.ubo.get_data(&view_proj);

        self.ubo_buffers[index].copy_data_into_buffer(device, &model_data);
    }

    fn descriptor_set(&self, image_index: usize) -> vk::DescriptorSet {
        self.descriptor_sets[image_index]
    }
    
    fn vertex_buffer(&self) -> BufferWrapper {
        self.vertex_buffer
    }
    
    fn index_buffer(&self) -> BufferWrapper {
        self.index_buffer
    }

    fn index_len(&self) -> u32 {
        self.indices.len() as u32
    }
    
    fn name(&self) -> String {
        self.name.clone()
    }
}

impl Hash for ObjectPrototype {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.vertices.hash(state);
        self.indices.hash(state);
        self.image.hash(state);
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
