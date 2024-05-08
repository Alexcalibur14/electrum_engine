#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use glam::Mat4;
use vulkanalia::prelude::v1_2::*;

use crate::{RenderStats, RendererData};

pub mod mesh;
pub mod vertices;

pub trait Renderable {
    unsafe fn draw(
        &self,
        instance: &Instance,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ModelMVP {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

impl ModelMVP {
    pub fn get_data(&self, view_proj: &Mat4) -> ModelData {
        ModelData {
            model: self.model,
            mvp: *view_proj * self.model,
            normal: self.model.inverse().transpose(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ModelData {
    model: Mat4,
    mvp: Mat4,
    normal: Mat4,
}
