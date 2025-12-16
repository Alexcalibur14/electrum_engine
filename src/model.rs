use ash::{Device, Instance, vk};

use crate::{RendererData, Vertex, buffer::Buffer};

pub struct MeshData<'a> {
    name: &'a str,
    vertex_buffer: Option<Buffer>,
    index_type: Option<vk::IndexType>,
    index_buffer: Option<Buffer>,
    instance_buffer: Option<Buffer>,
}

impl<'a> MeshData<'a> {
    pub fn new(name: &'a str) -> Self {
        MeshData {
            name,
            vertex_buffer: None,
            index_type : None,
            index_buffer: None,
            instance_buffer: None
        }
    }

    pub fn vertex_buffer(&self) -> Option<Buffer> {
        self.vertex_buffer
    }

    pub fn index_buffer(&self) -> Option<Buffer> {
        self.index_buffer
    }

    pub fn index_type(&self) -> Option<vk::IndexType> {
        self.index_type
    }

    pub fn instance_buffer(&self) -> Option<Buffer> {
        self.instance_buffer
    }

    /// Creates a buffer with [HOST_COHERENT](vk::MemoryPropertyFlags::HOST_COHERENT) and [HOST_VISIBLE](vk::MemoryPropertyFlags::HOST_VISIBLE)
    /// as memory properties and fills it with the [vertices](Self::vertices). \ Recomended only for vertex data that changes often
    pub fn build_vertex_host<V: Vertex>(&mut self, instance: &Instance, device: &Device, data: &RendererData, vertices: &[V]) {
        let vertex_buffer = Buffer::create_and_load(
            instance,
            device,
            data,
            vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &format!("{} Vertex Buffer", self.name)
        );

        self.vertex_buffer = Some(vertex_buffer);
    }

    /// Creates a buffer with [HOST_COHERENT](vk::MemoryPropertyFlags::HOST_COHERENT) and [HOST_VISIBLE](vk::MemoryPropertyFlags::HOST_VISIBLE)
    /// as memory properties and fills it with the [indices](Self::indices). \ Recomended only for index data that changes often
    pub fn build_index_host<I>(&mut self, instance: &Instance, device: &Device, data: &RendererData, indices: &[I], index_type: vk::IndexType) {
        self.index_buffer = Some(Buffer::create_and_load(
            instance,
            device,
            data,
            indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &format!("{} Index Buffer", self.name)
        ));

        self.index_type = Some(index_type)
    }

    /// Creates a buffer with [HOST_COHERENT](vk::MemoryPropertyFlags::HOST_COHERENT) and [HOST_VISIBLE](vk::MemoryPropertyFlags::HOST_VISIBLE)
    /// as memory properties and fills it with the [instances](Self::instances). \ Recomended only for instance data that changes often
    pub fn build_instance_host<I>(&mut self, instance: &Instance, device: &Device, data: &RendererData, instances: &[I]) {
        self.instance_buffer = Some(Buffer::create_and_load(
            instance,
            device,
            data,
            instances,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &format!("{} Instance Buffer", self.name)
        ));
    }

    /// Creates a buffer with [DEVICE_LOCAL](vk::MemoryPropertyFlags::DEVICE_LOCAL)
    /// as memory properties and fills it with the [vertices](Self::vertices). \ Recomended only for static vertex data
    pub fn build_vertex_staged<V: Vertex>(&mut self, instance: &Instance, device: &Device, data: &RendererData, vertices: &[V]) {
        self.vertex_buffer = Some(Buffer::create_and_stage(
            instance,
            device,
            data,
            vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &format!("{} Vertex Buffer", self.name)
        ));
    }

    /// Creates a buffer with [DEVICE_LOCAL](vk::MemoryPropertyFlags::DEVICE_LOCAL)
    /// as memory properties and fills it with the [indices](Self::indices). \ Recomended only for static index data
    pub fn build_index_staged<I>(&mut self, instance: &Instance, device: &Device, data: &RendererData, indices: &[I], index_type: vk::IndexType) {
        self.index_buffer = Some(Buffer::create_and_stage(
            instance,
            device,
            data,
            indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &format!("{} Index Buffer", self.name)
        ));

        self.index_type = Some(index_type);
    }

    /// Creates a buffer with [DEVICE_LOCAL](vk::MemoryPropertyFlags::DEVICE_LOCAL)
    /// as memory properties and fills it with the [instances](Self::instances). \ Recomended only for static instance data
    pub fn build_instance_staged<I>(&mut self, instance: &Instance, device: &Device, data: &RendererData, instances: &[I]) {
        self.instance_buffer = Some(Buffer::create_and_stage(
            instance,
            device,
            data,
            instances,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &format!("{} Instance Buffer", self.name)
        ));
    }

    /// Binds the Vertex, Index, and Instance buffers (if present) \
    /// Does nothing if the buffers have not been built
    pub fn bind_buffers(&self, device: &Device, command_buffer: vk::CommandBuffer) {
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

    pub fn destroy(&mut self, device: &Device) {
        if let Some(vertex_buffer) = self.vertex_buffer.as_mut() {
            vertex_buffer.destroy(device);
        }
        self.vertex_buffer = None;

        if let Some(index_buffer) = self.index_buffer.as_mut() {
            index_buffer.destroy(device);
        }
        self.index_buffer = None;

        if let Some(instance_buffer) = self.instance_buffer.as_mut() {
            instance_buffer.destroy(device);
        }
        self.instance_buffer = None;
    }
}
