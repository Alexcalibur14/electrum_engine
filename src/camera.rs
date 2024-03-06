use std::{mem::size_of, ptr::copy_nonoverlapping as memcpy};

use glam::{Mat4, Quat, Vec3};
use vulkanalia::prelude::v1_2::*;

use crate::buffer::{create_buffer, BufferWrapper};

pub trait Camera {
    /// Returns the View matrix
    fn view(&self) -> Mat4;
    /// Returns the Projection matrix
    fn proj(&self) -> Mat4;
    /// Returns the Inverse Projection matrix
    fn inv_proj(&self) -> Mat4;

    /// calculate the view matrix
    fn calculate_view(&mut self, device: &Device);
    /// calculate the projection matrix
    fn calculate_proj(&mut self, device: &Device);
    /// calculate the sets the aspect for the projection matrix
    fn set_aspect(&mut self, aspect_ratio: f32);

    fn view_buffer(&self, image_index: usize) -> BufferWrapper;
    fn proj_buffer(&self, image_index: usize) -> BufferWrapper;

    fn destroy(&self, device: &Device);
}

#[derive(Debug, Clone, Default)]
pub struct SimpleCamera {
    pub position: Vec3,
    pub rotation: Vec3,
    pub view: Mat4,
    pub projection: Projection,

    view_buffers: Vec<BufferWrapper>,
    proj_buffers: Vec<BufferWrapper>,
}

impl SimpleCamera {
    pub fn new(instance: &Instance, device: &Device, data: &crate::AppData, position: Vec3, rotation: Vec3, projection: Projection) -> Self {
        let view = Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, rotation.x, rotation.y, rotation.z), position);

        let mut view_buffers = vec![];
        let mut proj_buffers = vec![];

        for _ in 0..data.swapchain_images.len() {
            let view_buffer = unsafe { create_buffer(
                instance,
                device,
                data,
                size_of::<Mat4>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE
            ) }.unwrap();

            let view_mem = unsafe { device.map_memory(view_buffer.memory, 0, size_of::<Mat4>() as u64, vk::MemoryMapFlags::empty()) }.unwrap();

            unsafe { memcpy(&view, view_mem.cast(), 1) };

            unsafe { device.unmap_memory(view_buffer.memory) };

            view_buffers.push(view_buffer);

            let proj_buffer = unsafe { create_buffer(
                instance,
                device,
                data,
                size_of::<Mat4>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE
            ) }.unwrap();

            let proj_mem = unsafe { device.map_memory(proj_buffer.memory, 0, size_of::<Mat4>() as u64, vk::MemoryMapFlags::empty()) }.unwrap();

            unsafe { memcpy(&view, proj_mem.cast(), 1) };

            unsafe { device.unmap_memory(proj_buffer.memory) };

            proj_buffers.push(proj_buffer);
        }
        
        SimpleCamera {
            position,
            rotation,
            view,
            projection,
            view_buffers,
            proj_buffers,
        }
    }

}

impl Camera for SimpleCamera {
    fn view(&self) -> Mat4 {
        self.view
    }

    fn proj(&self) -> Mat4 {
        self.projection.proj
    }

    fn inv_proj(&self) -> Mat4 {
        self.projection.inv_proj
    }

    fn calculate_view(&mut self, device: &Device) {
        self.view = Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, self.rotation.x, self.rotation.y, self.rotation.z), self.position);

        self.view_buffers.iter().for_each(|b| {
            let view_mem = unsafe { device.map_memory(b.memory, 0, size_of::<Mat4>() as u64, vk::MemoryMapFlags::empty()) }.unwrap();

            unsafe { memcpy(&self.view, view_mem.cast(), 1) };

            unsafe { device.unmap_memory(b.memory) };
        });
    }

    fn calculate_proj(&mut self, device: &Device) {
        self.projection.recalculate();
        
        self.proj_buffers.iter().for_each(|b| {
            let proj_mem = unsafe { device.map_memory(b.memory, 0, size_of::<Mat4>() as u64, vk::MemoryMapFlags::empty()) }.unwrap();

            unsafe { memcpy(&self.projection.proj, proj_mem.cast(), 1) };

            unsafe { device.unmap_memory(b.memory) };
        });
    }

    fn set_aspect(&mut self, aspect_ratio: f32) {
        self.projection.aspect_ratio = aspect_ratio;
    }
    
    fn view_buffer(&self, image_index: usize) -> BufferWrapper {
        self.view_buffers[image_index]
    }
    
    fn proj_buffer(&self, image_index: usize) -> BufferWrapper {
        self.proj_buffers[image_index]
    }
    
    fn destroy(&self, device: &Device) {
        self.view_buffers.iter().for_each(|b| b.destroy(device));
        self.proj_buffers.iter().for_each(|b| b.destroy(device));
    }
}


#[derive(Debug, Clone, Default)]
pub struct Projection {
    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub proj: Mat4,
    pub inv_proj: Mat4,
}

impl Projection {
    pub fn new(fov_y_rad: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let proj = Mat4::perspective_rh(fov_y_rad, aspect_ratio, z_near, z_far);
        let inv_proj = proj.inverse();
        Projection {
            fov_y_rad,
            aspect_ratio,
            z_near,
            z_far,
            proj,
            inv_proj
        }
    }

    pub fn recalculate(&mut self) -> &mut Self {
        let proj = Mat4::perspective_rh(self.fov_y_rad, self.aspect_ratio, self.z_near, self.z_far);
        self.proj = proj;
        self.inv_proj = proj.inverse();
        self
    }
}
