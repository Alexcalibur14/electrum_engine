use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_2::*;

use anyhow::{anyhow, Result};

use crate::{begin_command_label, end_command_label, set_object_name, RendererData};

pub unsafe fn begin_single_time_commands(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    name: String,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    begin_command_label(instance, command_buffer, name, [1.0, 0.0, 1.0, 1.0]);
    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

pub unsafe fn end_single_time_commands(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    device.end_command_buffer(command_buffer)?;
    end_command_label(instance, command_buffer);

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}

pub unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
    name: Option<String>,
) -> Result<BufferWrapper> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(buffer);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    let wrapper = BufferWrapper {
        buffer,
        memory: buffer_memory,
        size,
    };

    if let Some(object_name) = name {
        set_object_name(
            instance,
            device,
            object_name.clone(),
            vk::ObjectType::BUFFER,
            buffer.as_raw(),
        )?;

        set_object_name(
            instance,
            device,
            object_name + " Memory",
            vk::ObjectType::DEVICE_MEMORY,
            buffer_memory.as_raw(),
        )?;
    }

    Ok(wrapper)
}

pub unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &RendererData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(data.physical_device);

    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];

            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

/// Copies the contents of one buffer into another buffer
pub fn copy_buffer(
    instance: &Instance,
    device: &Device,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
    data: &RendererData,
) -> Result<()> {
    unsafe {
        let command_buffer =
            begin_single_time_commands(instance, device, data, "Copy Buffer".to_string())?;

        let regions = vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(size);
        device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

        end_single_time_commands(instance, device, data, command_buffer)?;
    }
    
    Ok(())
}

pub fn create_and_stage_buffer<T>(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    name: Option<String>,
    bytes: &Vec<T>,
) -> Result<BufferWrapper> {
    let staging_buffer = unsafe { create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        None
    ) }.unwrap();

    staging_buffer.copy_vec_into_buffer(device, bytes);

    let buffer = unsafe { create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        name
    ) }.unwrap();

    copy_buffer(instance, device, staging_buffer.buffer, buffer.buffer, size.into(), data)?;

    staging_buffer.destroy(device);

    Ok(buffer)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BufferWrapper {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    size: u64
}

impl BufferWrapper {
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }

    /// Copies data into the buffer, if you are trying to copy a vec use `copy_vec_into_buffer`
    pub fn copy_data_into_buffer<T>(&self, device: &Device, data: &T) {
        let buffer_mem = unsafe {
            device.map_memory(
                self.memory,
                0,
                self.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .unwrap();

        unsafe { memcpy(data, buffer_mem.cast(), 1) };

        unsafe { device.unmap_memory(self.memory) };
    }

    /// Copies data into the buffer with an offset in the buffer, if you are trying to copy a vec use `copy_vec_into_buffer_with_offset`
    pub fn copy_data_into_buffer_with_offset<T>(&self, device: &Device, data: &T, offset: u64, size: u64) {
        let buffer_mem = unsafe {
            device.map_memory(
                self.memory,
                offset,
                size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .unwrap();

        unsafe { memcpy(data, buffer_mem.cast(), 1) };

        unsafe { device.unmap_memory(self.memory) };
    }

    /// Copies a vec into the buffer
    pub fn copy_vec_into_buffer<T> (&self, device: &Device, data: &Vec<T>) {
        let buffer_mem = unsafe {
            device.map_memory(
                self.memory,
                0,
                self.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .unwrap();

        unsafe { memcpy(data.as_ptr(), buffer_mem.cast(), data.len()) };

        unsafe { device.unmap_memory(self.memory) };
    }

    /// Copies a vec into the buffer with an offset of `offset` in the buffer
    pub fn copy_vec_into_buffer_with_offset<T> (&self, device: &Device, data: &Vec<T>, offset: u64, size: u64) {
        let buffer_mem = unsafe {
            device.map_memory(
                self.memory,
                offset,
                size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .unwrap();

        unsafe { memcpy(data.as_ptr(), buffer_mem.cast(), data.len()) };

        unsafe { device.unmap_memory(self.memory) };
    }
}
