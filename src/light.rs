use std::{mem::size_of, ptr::copy_nonoverlapping as memcpy};

use glam::Vec3;
use anyhow::Result;
use vulkanalia::prelude::v1_2::*;

use crate::{buffer::{create_buffer, BufferWrapper}, RendererData};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PointLight {
    pub position: Vec3,
    _pad1: u32,
    pub colour: Vec3,
    pub strength: f32,
}

impl PointLight {
    pub fn new(position: Vec3, colour: Vec3, strength: f32) -> PointLight {
        PointLight {
            position,
            _pad1: 0,
            colour,
            strength,
        }
    }

    pub fn get_buffers(&self, instance: &Instance, device: &Device, data: &RendererData ) -> Result<Vec<BufferWrapper>> {
        let mut buffers = vec![];
        for _ in 0..data.swapchain_images.len() {
            let buffer = unsafe { create_buffer(instance, device, data, size_of::<PointLight>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE) }?;
            
            let mem = unsafe { device.map_memory(buffer.memory, 0, size_of::<PointLight>() as u64, vk::MemoryMapFlags::empty()) }?;

            unsafe { memcpy(self, mem.cast(), 1) };

            unsafe { device.unmap_memory(buffer.memory) };
            buffers.push(buffer);
        }

        Ok(buffers)
    }
}
