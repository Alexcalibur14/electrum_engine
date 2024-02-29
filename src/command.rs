use vulkanalia::prelude::v1_2::*;

use anyhow::Result;

use crate::{AppData, QueueFamilyIndices};


pub unsafe fn create_command_pool(instance: &Instance, device: &Device, data: &AppData) -> Result<vk::CommandPool> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(indices.graphics);

    Ok(device.create_command_pool(&info, None)?)
}

pub unsafe fn create_command_buffers(device: &Device, data: &AppData) -> Result<Vec<vk::CommandBuffer>> {

    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    Ok(device.allocate_command_buffers(&allocate_info)?)
}