#![allow(clippy::too_many_arguments)]

use glam::Mat4;

use ash::vk;
use ash::{Device, Instance};

use crate::{create_and_stage_buffer, BufferWrapper, RendererData};

pub mod mesh;
pub mod vertices;

pub trait Vertex {
    fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription>;
    fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelMVP {
    model: Mat4,
}

impl ModelMVP {
    pub fn get_data(&self) -> ModelData {
        ModelData {
            model: self.model,
            normal: self.model.inverse().transpose(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelData {
    model: Mat4,
    normal: Mat4,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshData {
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertex_buffer: BufferWrapper,
    pub index_buffer: BufferWrapper,
}

impl MeshData {
    pub fn new<T>(instance: &Instance, device: &Device, data: &RendererData, vertices: &[T], indices: &[u32], name: &str) -> Self {
        let vertex_buffer = create_and_stage_buffer(
            instance,
            device,
            data,
            (size_of::<T>() * vertices.len()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &format!("{} Vertex Buffer", name),
            vertices,
        ).unwrap();

        let vertex_count = vertices.len() as u32;

        let index_buffer = create_and_stage_buffer(
            instance,
            device,
            data,
            (size_of::<u32>() * indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &format!("{} Index Buffer", name),
            indices,
        ).unwrap();

        let index_count = indices.len() as u32;
        
        MeshData {
            vertex_count,
            index_count,
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn destroy(&self, device: &Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            vertex_count: 0,
            index_count: 0,
            vertex_buffer: BufferWrapper::default(),
            index_buffer: BufferWrapper::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DescriptorManager {
    pub descriptors: Vec<Vec<vk::DescriptorSet>>,
    pub subpasses: Vec<(u32, Vec<(u32, Vec<usize>)>)>,
}

impl DescriptorManager {
    fn get_descriptors(&self, render_pass: u32, subpass: u32, image_index: usize) -> Vec<vk::DescriptorSet> {
        let ids = &self.subpasses.iter().find(|(s, _)| *s == render_pass).unwrap().1.iter().find(|(s, _)| *s == subpass).unwrap().1;
        let mut descriptors = Vec::with_capacity(ids.len());
        for id in ids {
            descriptors.push(self.descriptors[image_index][*id]);
        }

        descriptors
    }
}
