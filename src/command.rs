use ash::vk;
use ash::{Entry, Device, Instance};

use crate::{begin_command_label, end_command_label, QueueFamilyIndices, RendererData};
use anyhow::Result;

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


///
/// # Safety
/// the function `end_single_time_commands` must be called after this
/// function is called
pub unsafe fn begin_single_time_commands(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    name: &str,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    let info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    begin_command_label(instance, device, command_buffer, name, [1.0, 0.0, 1.0, 1.0]);
    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

///
/// # Safety
/// this function **must** only be called after a call to `begin_single_time_commands`
pub unsafe fn end_single_time_commands(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    device.end_command_buffer(command_buffer)?;
    end_command_label(instance, device, command_buffer);

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::default().command_buffers(command_buffers);

    device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}

pub fn execute_command<F>(instance: &Instance, device: &Device, data: &RendererData, name: &str, f: F)
where F: FnOnce(vk::CommandBuffer)
{
    let command_buffer = unsafe { begin_single_time_commands(instance, device, data, name) }.unwrap();
    f(command_buffer);
    unsafe { end_single_time_commands(instance, device, data, command_buffer) }.unwrap();
}
