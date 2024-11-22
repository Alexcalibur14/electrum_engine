use std::{hash::Hash, mem::size_of};

use glam::Vec3;
use vulkanalia::prelude::v1_2::*;

use crate::{
    buffer::{create_buffer, BufferWrapper}, DescriptorBuilder, RendererData
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PointLight {
    pub position: Vec3,
    _pad1: u32,
    pub colour: Vec3,
    pub strength: f32,
}

impl Hash for PointLight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.position[0].to_bits().hash(state);
        self.position[1].to_bits().hash(state);
        self.position[2].to_bits().hash(state);
        self.colour[0].to_bits().hash(state);
        self.colour[1].to_bits().hash(state);
        self.colour[2].to_bits().hash(state);
        self.strength.to_bits().hash(state);
    }
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
}

#[derive(Debug, Clone, Default)]
pub struct LightGroup {
    point_lights: Vec<u64>,

    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub buffers: Vec<BufferWrapper>,
}

impl LightGroup {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,
        mut loaded_lights: Vec<u64>,
        capacity: usize,
    ) -> Self {
        let mut lights = Vec::with_capacity(capacity);
        lights.append(&mut loaded_lights);

        let light_data = lights
            .iter()
            .map(|i| *data.point_light_data.get(i).unwrap())
            .collect::<Vec<PointLight>>();

        let mut buffers = vec![];

        for i in 0..data.swapchain_images.len() {
            let buffer = unsafe {
                create_buffer(
                    instance,
                    device,
                    data,
                    (size_of::<PointLight>() * capacity) as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    Some(format!("Light Buffer {}", i)),
                )
            }
            .unwrap();

            buffer.copy_vec_into_buffer(device, &light_data);

            buffers.push(buffer);
        }

        let mut descriptor_sets = vec![];
        let mut descriptor_set_layout= Default::default();

        for i in 0..data.swapchain_images.len() {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(buffers[i].buffer)
                .offset(0)
                .range((size_of::<PointLight>() * capacity) as u64)
                .build();

            let (set, set_layout) = DescriptorBuilder::new()
                .bind_buffer(0, capacity as u32, &[buffer_info], vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::FRAGMENT)
                .build(device, &mut data.global_layout_cache, &mut data.global_descriptor_pools).unwrap();

            descriptor_sets.push(set);
            descriptor_set_layout = set_layout;
        }

        LightGroup {
            point_lights: lights,
            descriptor_set_layout,
            descriptor_sets,
            buffers,
        }
    }

    pub fn add_light(&mut self, device: &Device, data: &RendererData, light: u64) {
        let offset = (self.point_lights.len() * size_of::<PointLight>()) as u64;
        self.point_lights.push(light);

        let light_data = data.point_light_data.get(&light).unwrap();

        for buffer in self.buffers.clone() {
            buffer.copy_data_into_buffer_with_offset(
                device,
                &light_data,
                offset,
                size_of::<PointLight>() as u64,
            );
        }
    }

    pub fn get_lights(&self) -> Vec<u64> {
        self.point_lights.clone()
    }

    pub fn get_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn get_descriptor_sets(&self) -> Vec<vk::DescriptorSet> {
        self.descriptor_sets.clone()
    }

    pub fn destroy(&self, device: &Device) {
        self.buffers.iter().for_each(|b| b.destroy(device));
    }
}

impl Hash for LightGroup {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.point_lights.hash(state);
        self.descriptor_sets.hash(state);
    }
}
