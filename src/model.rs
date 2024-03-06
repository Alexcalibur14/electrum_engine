use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::ptr::copy_nonoverlapping as memcpy;
use std::{hash::Hasher, mem::size_of};
use std::hash::Hash;

use glam::{vec2, vec3, Vec2, Vec3};
use anyhow::Result;
use vulkanalia::prelude::v1_2::*;

use crate::buffer::{copy_buffer, create_buffer, BufferWrapper};
use crate::shader::{Material, PipelineMeshSettings, VFShader};
use crate::AppData;


pub trait Renderable {
    unsafe fn draw(&self, device: &Device, command_buffer: vk::CommandBuffer, image_index: usize);
    // fn update(&mut self, );
    fn destroy_swapchain(&self, device: &Device);
    fn recreate_swapchain(&mut self, device: &Device, data: &AppData);
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

#[derive(Clone)]
pub struct ObjectPrototype {
    vertices: Vec<PCTVertex>,
    indices: Vec<u32>,

    vertex_buffer: BufferWrapper,
    index_buffer: BufferWrapper,

    material: Material,
}

impl Renderable for ObjectPrototype {
    unsafe fn draw(&self, device: &Device, command_buffer: vk::CommandBuffer, _image_index: usize) {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.material.pipeline);
        device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer.buffer], &[0]);
        device.cmd_bind_index_buffer(command_buffer, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
        device.cmd_draw_indexed(command_buffer, self.indices.len() as u32, 1, 0, 0, 0);
    }
    
    fn destroy_swapchain(&self, device: &Device) {
        self.material.destroy_swapchain(device);
    }

    fn recreate_swapchain(&mut self, device: &Device, data: &AppData) {
        self.material.recreate_swapchain(device, data);
    }

    fn destroy(&self, device: &Device) {
        self.material.destroy(device);
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }

    fn clone_dyn(&self) -> Box<dyn Renderable> {
        Box::new(self.clone())
    }
}


impl ObjectPrototype {
    pub fn load(path: &str, device: &Device, data: &AppData, shader: VFShader) -> Self {
        let (vertices, indices) = load_model_temp(path).unwrap();

        let mesh_settings = PipelineMeshSettings {
            binding_descriptions: PCTVertex::binding_descriptions(),
            attribute_descriptions: PCTVertex::attribute_descriptions(),
            front_face: vk::FrontFace::CLOCKWISE,
            ..Default::default()
        };

        let material = Material::new(device, data, vec![], vec![], shader, mesh_settings);

        ObjectPrototype {
            vertices,
            indices,

            vertex_buffer: BufferWrapper::default(),
            index_buffer: BufferWrapper::default(),

            material,
        }
    }

    pub unsafe fn generate_vertex_buffer(&mut self, instance: &Instance, device: &Device, data: &AppData) {
        let size = (size_of::<PCTVertex>() * self.vertices.len()) as u64;

        let staging_buffer = create_buffer(
                instance,
                device,
                data,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ).unwrap();

        let memory = device.map_memory(
            staging_buffer.memory,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        ).unwrap();

        memcpy(self.vertices.as_ptr(), memory.cast(), self.vertices.len());

        device.unmap_memory(staging_buffer.memory);

        let vertex_buffer = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).unwrap();

        copy_buffer(device, staging_buffer.buffer, vertex_buffer.buffer, size, data).unwrap();

        device.destroy_buffer(staging_buffer.buffer, None);
        device.free_memory(staging_buffer.memory, None);

        self.vertex_buffer = vertex_buffer;
    }

    pub unsafe fn generate_index_buffer(&mut self, instance: &Instance, device: &Device, data: &AppData) {
        let size = (size_of::<u32>() * self.indices.len()) as u64;

        let staging_buffer = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        ).unwrap();

        let memory = device.map_memory(
            staging_buffer.memory,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        ).unwrap();

        memcpy(self.indices.as_ptr(), memory.cast(), self.indices.len());

        device.unmap_memory(staging_buffer.memory);

        let index_buffer = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).unwrap();

        copy_buffer(device,  staging_buffer.buffer, index_buffer.buffer, size, data).unwrap();

        device.destroy_buffer(staging_buffer.buffer, None);
        device.free_memory(staging_buffer.memory, None);

        self.index_buffer = index_buffer;
    }
}

fn load_model_temp(path: &str) -> Result<(Vec<PCTVertex>, Vec<u32>)> {
    let mut reader = BufReader::new(File::open(path)?);

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions { triangulate: false, ..Default::default() },
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
                color: vec3(1.0, 1.0, 1.0),
                tex_coord: vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
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
    pub color: Vec3,
    pub tex_coord: Vec2,
}

impl Vertex for PCTVertex {
    fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![
            vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(size_of::<PCTVertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build()
        ]
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
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec3>() as u32)
                .build(),

            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
                .build(),
        ]
    }
}

impl PartialEq for PCTVertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.color == other.color
            && self.tex_coord == other.tex_coord
    }
}

impl Eq for PCTVertex {}

impl Hash for PCTVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}
