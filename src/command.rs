use ash::{vk, Entry};
use ash::{Device, Instance};

use anyhow::Result;

use crate::{QueueFamilyIndices, RendererData};

pub(crate) unsafe fn create_command_pools(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &mut RendererData,
) -> Result<()> {
    data.command_pool = create_command_pool(entry, instance, device, data)?;

    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = create_command_pool(entry, instance, device, data)?;
        data.command_pools.push(command_pool);
    }

    Ok(())
}

unsafe fn create_command_pool(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &mut RendererData,
) -> Result<vk::CommandPool> {
    let indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(indices.graphics);

    Ok(device.create_command_pool(&info, None)?)
}

pub(crate) unsafe fn create_command_buffers(
    device: &Device,
    data: &mut RendererData,
) -> Result<()> {
    let num_images = data.swapchain_images.len();
    for image_index in 0..num_images {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(data.command_pools[image_index])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
        data.command_buffers.push(command_buffer);
    }

    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

    Ok(())
}
