pub mod primitives;

use core::fmt;
use std::path::Path;

use ash::vk;
use glam::{Vec2, Vec3};

use crate::{RendererData, RenderingDevice, Vertex, buffer::{Buffer, BufferType}, image::Image, resources::NamedVec};

#[derive(Debug, Clone, Default)]
pub struct MeshData {
    name: String,
    vertex_buffer: Option<Buffer>,
    vertex_len: u32,
    index_type: Option<vk::IndexType>,
    index_buffer: Option<Buffer>,
    index_len: u32,
    instance_buffer: Option<Buffer>,
    instance_len: u32,
}

impl MeshData {
    pub fn new(name: &str) -> Self {
        MeshData {
            name: name.to_owned(),
            vertex_buffer: None,
            vertex_len: 0,
            index_type : None,
            index_buffer: None,
            index_len: 0,
            instance_buffer: None,
            instance_len: 1,
        }
    }

    pub fn vertex_buffer(&self) -> Option<Buffer> {
        self.vertex_buffer
    }

    pub fn vertex_len(&self) -> u32 {
        self.vertex_len
    }

    pub fn index_buffer(&self) -> Option<Buffer> {
        self.index_buffer
    }

    pub fn index_type(&self) -> Option<vk::IndexType> {
        self.index_type
    }

    pub fn index_len(&self) -> u32 {
        self.index_len
    }

    pub fn instance_buffer(&self) -> Option<Buffer> {
        self.instance_buffer
    }

    pub fn instance_len(&self) -> u32 {
        self.instance_len
    }

    /// Creates a buffer with [HOST_COHERENT](vk::MemoryPropertyFlags::HOST_COHERENT) and [HOST_VISIBLE](vk::MemoryPropertyFlags::HOST_VISIBLE)
    /// as memory properties and fills it with the [vertices](Self::vertices). \ Recomended only for vertex data that changes often
    pub fn build_vertex_host<V: Vertex>(&mut self, device: &RenderingDevice, data: &RendererData, vertices: &[V]) {
        self.vertex_len = vertices.len() as u32;
        
        let vertex_buffer = Buffer::create_and_load(
            device,
            data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferType::HostLocal,
            vertices,
            &format!("{} Vertex", self.name)
        ).unwrap();

        self.vertex_buffer = Some(vertex_buffer);
    }

    /// Creates a buffer with [HOST_COHERENT](vk::MemoryPropertyFlags::HOST_COHERENT) and [HOST_VISIBLE](vk::MemoryPropertyFlags::HOST_VISIBLE)
    /// as memory properties and fills it with the [indices](Self::indices). \ Recomended only for index data that changes often
    pub fn build_index_host<I>(&mut self, device: &RenderingDevice, data: &RendererData, indices: &[I], index_type: vk::IndexType) {
        self.index_len = indices.len() as u32;
        
        self.index_buffer = Some(Buffer::create_and_load(
            device,
            data,
            vk::BufferUsageFlags::INDEX_BUFFER,
            BufferType::HostLocal,
            indices,
            &format!("{} Index", self.name)
        ).unwrap());

        self.index_type = Some(index_type)
    }

    /// Creates a buffer with [HOST_COHERENT](vk::MemoryPropertyFlags::HOST_COHERENT) and [HOST_VISIBLE](vk::MemoryPropertyFlags::HOST_VISIBLE)
    /// as memory properties and fills it with the [instances](Self::instances). \ Recomended only for instance data that changes often
    pub fn build_instance_host<I>(&mut self, device: &RenderingDevice, data: &RendererData, instances: &[I]) {
        self.instance_len = instances.len() as u32;
        
        self.instance_buffer = Some(Buffer::create_and_load(
            device,
            data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferType::HostLocal,
            instances,
            &format!("{} Instance", self.name)
        ).unwrap());
    }

    /// Creates a buffer with [DEVICE_LOCAL](vk::MemoryPropertyFlags::DEVICE_LOCAL)
    /// as memory properties and fills it with the [vertices](Self::vertices). \ Recomended only for static vertex data
    pub fn build_vertex_staged<V: Vertex>(&mut self, device: &RenderingDevice, data: &RendererData, vertices: &[V]) {
        self.vertex_len = vertices.len() as u32;

        self.vertex_buffer = Some(Buffer::create_and_load(
            device,
            data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferType::DeviceLocal,
            vertices,
            &format!("{} Vertex", self.name)
        ).unwrap());
    }

    /// Creates a buffer with [DEVICE_LOCAL](vk::MemoryPropertyFlags::DEVICE_LOCAL)
    /// as memory properties and fills it with the [indices](Self::indices). \ Recomended only for static index data
    pub fn build_index_staged<I>(&mut self, device: &RenderingDevice, data: &RendererData, indices: &[I], index_type: vk::IndexType) {
        self.index_len = indices.len() as u32;
        
        self.index_buffer = Some(Buffer::create_and_load(
            device,
            data,
            vk::BufferUsageFlags::INDEX_BUFFER,
            BufferType::DeviceLocal,
            indices,
            &format!("{} Index", self.name)
        ).unwrap());

        self.index_type = Some(index_type);
    }

    /// Creates a buffer with [DEVICE_LOCAL](vk::MemoryPropertyFlags::DEVICE_LOCAL)
    /// as memory properties and fills it with the [instances](Self::instances). \ Recomended only for static instance data
    pub fn build_instance_staged<I>(&mut self, device: &RenderingDevice, data: &RendererData, instances: &[I]) {
        self.instance_len = instances.len() as u32;
        
        self.instance_buffer = Some(Buffer::create_and_load(
            device,
            data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferType::DeviceLocal,
            instances,
            &format!("{} Instance", self.name)
        ).unwrap());
    }

    /// Binds the Vertex, Index, and Instance buffers (if present) \
    /// Does nothing if the buffers have not been built
    pub fn bind_buffers(&self, device: &RenderingDevice, command_buffer: vk::CommandBuffer) {
        if let Some(vertex_buffer) = self.vertex_buffer {
            unsafe { device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer.buffer()], &[0]) };
        }
        
        if let Some(index_buffer) = self.index_buffer {
            unsafe { device.cmd_bind_index_buffer(
                command_buffer,
                index_buffer.buffer(),
                0,
                *self.index_type.as_ref().unwrap()
            ); }
        }

        if let Some(instance_buffer) = self.instance_buffer {
            unsafe { device.cmd_bind_vertex_buffers(command_buffer, 0, &[instance_buffer.buffer()], &[0]); }
        }
    }

    pub fn destroy(&mut self, device: &RenderingDevice) {
        if let Some(vertex_buffer) = self.vertex_buffer.as_mut() {
            vertex_buffer.destroy(device);
        }
        self.vertex_buffer = None;
        self.vertex_len = 0;

        if let Some(index_buffer) = self.index_buffer.as_mut() {
            index_buffer.destroy(device);
        }
        self.index_buffer = None;
        self.index_len = 0;

        if let Some(instance_buffer) = self.instance_buffer.as_mut() {
            instance_buffer.destroy(device);
        }
        self.instance_buffer = None;
        self.instance_len = 0;
    }
}


pub fn basic_obj_loader<P: AsRef<Path> + fmt::Debug>(device: &RenderingDevice, data: &RendererData, path: P) -> Vec<MeshData> {
    let (models, _) = tobj::load_obj(path, &tobj::LoadOptions {
        single_index: true,
        triangulate: false,
        ignore_points: true,
        ignore_lines: true,
    }).unwrap();

    models.iter().map(|model| {
        let name = model.name.clone();
        let indices = model.mesh.indices.clone();
        let (positions, _) = model.mesh.positions.as_chunks::<3>();
        let (normals, _) = model.mesh.normals.as_chunks::<3>();
        let (colours, _) = model.mesh.vertex_color.as_chunks::<3>();
        let (uvs, _) = model.mesh.texcoords.as_chunks::<2>();

        let mut vertices = vec![];

        for i in 0..positions.len() {
            let position = positions[i].into();
            let normal = if normals.len() > 0 { normals[i] } else { [0.0, 0.0, 0.0] }.into();
            let colour = if colours.len() > 0 { colours[i] } else { [0.0, 0.0, 0.0] }.into();
            let uv = if uvs.len() > 0 { uvs[i] } else { [0.0, 0.0] }.into();

            vertices.push(
                OBJVertex {
                    position,
                    normal,
                    colour,
                    uv,
                }
            );
        }


        let mut mesh_data = MeshData::new(&name);

        mesh_data.build_index_staged(device, data, &indices, vk::IndexType::UINT32);
        mesh_data.build_vertex_staged(device, data, &vertices);

        mesh_data
    }).collect::<Vec<MeshData>>()
}

#[repr(C)]
#[derive(Debug, Vertex)]
pub struct OBJVertex {
    #[vertex(format = "R32G32B32_SFLOAT")]
    pub position: Vec3,
    #[vertex(format = "R32G32B32_SFLOAT")]
    pub normal: Vec3,
    #[vertex(format = "R32G32B32_SFLOAT")]
    pub colour: Vec3,
    #[vertex(format = "R32G32_SFLOAT")]
    pub uv: Vec2,
}

#[derive(Clone, Default)]
pub struct Object<'a> {
    name: &'a str,
    mesh_data: MeshData,
    buffers: NamedVec<'a, Buffer>,
    images: NamedVec<'a, Image>,
    descriptor_sets: NamedVec<'a, vk::DescriptorSet>
}

impl<'a> Object<'a> {
    pub fn new(name: &'a str) -> Self {
        Object {
            name,
            mesh_data: MeshData::new(name),
            buffers: NamedVec::new(),
            images: NamedVec::new(),
            descriptor_sets: NamedVec::new(),
        }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn mesh_data(&self) -> &MeshData {
        &self.mesh_data
    }

    pub fn mesh_data_mut(&mut self) -> &mut MeshData {
        &mut self.mesh_data
    }

    pub fn buffers(&self) -> &NamedVec<'a, Buffer> {
        &self.buffers
    }

    pub fn images(&self) -> &NamedVec<'a, Image> {
        &self.images
    }

    pub fn descriptor_sets(&self) -> &NamedVec<'a, vk::DescriptorSet> {
        &self.descriptor_sets
    }

    pub fn get_buffer(&self, name: &'a str) -> &Buffer {
        self.buffers.get(name).unwrap()
    }

    pub fn get_image(&self, name: &'a str) -> &Image {
        self.images.get(name).unwrap()
    }

    pub fn get_descriptor_set(&self, name: &'a str) -> &vk::DescriptorSet {
        self.descriptor_sets.get(name).unwrap()
    }
}


impl<'a> Object<'a> {
    pub fn add_buffer(&mut self, buffer: Buffer, name: &'a str) {
        self.buffers.push(buffer, name);
    }

    pub fn add_image(&mut self, image: Image, name: &'a str) {
        self.images.push(image, name);
    }

    pub fn add_descriptor_set(&mut self, descriptor_set: vk::DescriptorSet, buffers: &[(Buffer, &'a str)], images: &[(Image, &'a str)], name: &'a str) {
        self.descriptor_sets.push(descriptor_set, name);
        buffers.iter().for_each(|(buffer, name)| self.buffers.push(*buffer, name));
        images.iter().for_each(|(image, name)| self.images.push(*image, name));
    }

    pub fn replace_descriptor_set(&mut self, descriptor_set: vk::DescriptorSet, name: &'a str) {
        *self.descriptor_sets.get_mut(name).unwrap() = descriptor_set;
    }

    pub fn destroy(&mut self, device: &RenderingDevice) {
        self.mesh_data.destroy(device);

        self.buffers.items_mut().iter_mut().for_each(|buffer| buffer.destroy(device));
        self.buffers.clear();

        self.images.items_mut().iter_mut().for_each(|image| image.destroy(device));
        self.images.clear();

        self.descriptor_sets.clear()
    }
}
