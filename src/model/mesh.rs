#![allow(dead_code)]
use std::{collections::HashMap, fs::File, io::BufReader, mem::size_of};

use anyhow::Result;
use glam::{vec2, vec3, Mat4, Vec3};
use vulkanalia::prelude::v1_2::*;

use crate::{
    buffer::{create_buffer, BufferWrapper}, vertices::PCTVertex, DescriptorBuilder, RenderStats, Renderable, RendererData
};

use super::{MeshData, ModelData, ModelMVP};

#[derive(Clone)]
pub struct Quad {
    name: String,

    points: [Vec3; 4],
    normal: Vec3,

    mesh_data: MeshData,

    image: usize,

    ubo: ModelMVP,
    ubo_buffers: Vec<BufferWrapper>,

    descriptor_sets: Vec<vk::DescriptorSet>,

    loaded: bool,
}

impl Quad {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,

        points: [Vec3; 4],
        normal: Vec3,

        image: usize,

        model: Mat4,
        view: Mat4,
        proj: Mat4,

        name: String,
    ) -> Self {
        let mut vertices = vec![];

        let uvs = [
            vec2(0.0, 0.0),
            vec2(1.0, 0.0),
            vec2(0.0, 1.0),
            vec2(1.0, 1.0),
        ];

        for (i, point) in points.iter().enumerate() {
            vertices.push(PCTVertex {
                pos: *point,
                tex_coord: uvs[i],
                normal,
            })
        }

        let indices: Vec<u32> = vec![0, 3, 1, 0, 2, 3];

        let mesh_data = MeshData::new(instance, device, data, &vertices, &indices, &name);

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
        
        let texture = data.textures.get(image).unwrap();
        let sampler = texture.sampler.unwrap();

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
                .sampler(sampler)
                .build();

            let (set, _) = DescriptorBuilder::new()
                .bind_buffer(0, 1, &[buffer_info], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
                .bind_image(1, 1, &[image_info], vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
                .build(device, &mut data.global_layout_cache, &mut data.global_descriptor_pools).unwrap();

            descriptor_sets.push(set);
        }

        Quad {
            name,
            mesh_data,
            image,
            ubo,
            ubo_buffers,
            descriptor_sets,
            points,
            normal,
            loaded: true,
        }
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
        self.mesh_data.destroy(device);
        self.ubo_buffers.iter().for_each(|b| b.destroy(device));
    }

    fn clone_dyn(&self) -> Box<dyn Renderable> {
        Box::new(self.clone())
    }
    
    fn descriptor_set(&self, image_index: usize) -> vk::DescriptorSet {
        self.descriptor_sets[image_index]
    }
    
    fn mesh_data(&self) -> &MeshData {
        &self.mesh_data
    }

    fn name(&self) -> String {
        self.name.clone()
    }
    
    fn loaded(&self) -> bool {
        self.loaded
    }
}

#[derive(Clone)]
pub struct ObjectPrototype {
    name: String,

    mesh_data: MeshData,

    ubo: ModelMVP,
    ubo_buffers: Vec<BufferWrapper>,

    image: usize,

    descriptor_sets: Vec<vk::DescriptorSet>,

    loaded: bool,
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
        image: usize,
        name: String,
    ) -> Self {
        let (vertices, indices) = load_model_temp(path).unwrap();

        let mesh_data = MeshData::new(instance, device, data, &vertices, &indices, &name);

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

        let texture = data.textures.get(image).unwrap();
        let sampler = texture.sampler.unwrap();

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
                .sampler(sampler)
                .build();

            let (set, _) = DescriptorBuilder::new()
                .bind_buffer(0, 1, &[buffer_info], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
                .bind_image(1, 1, &[image_info], vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
                .build(device, &mut data.global_layout_cache, &mut data.global_descriptor_pools).unwrap();

            descriptor_sets.push(set);
        }

        ObjectPrototype {
            name,

            mesh_data,

            ubo,
            ubo_buffers,

            descriptor_sets,
            image,

            loaded: true,
        }
    }
}

impl Renderable for ObjectPrototype {
    fn destroy_swapchain(&self, _device: &Device) {
    }

    fn recreate_swapchain(&mut self, _device: &Device, _data: &mut RendererData) {
    }

    fn destroy(&mut self, device: &Device) {
        self.mesh_data.destroy(device);
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
    
    fn mesh_data(&self) -> &MeshData {
        &self.mesh_data
    }

    fn name(&self) -> String {
        self.name.clone()
    }
    
    fn loaded(&self) -> bool {
        self.loaded
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
