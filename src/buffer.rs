use ash::vk::{self, Handle};
use ash::{Device, Instance};
use anyhow::{Result, anyhow};

use std::ptr::copy_nonoverlapping as memcpy;

use crate::{begin_single_time_commands, end_single_time_commands, set_object_name, RendererData};

pub fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
    name: &str,
) -> Result<Buffer> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_info = unsafe { vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(get_memory_type_index(
                instance,
                data,
                properties,
                requirements,
            )?) };

    let buffer_memory = unsafe { device.allocate_memory(&memory_info, None)? };

    unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };

    let wrapper = Buffer {
        buffer,
        memory: buffer_memory,
        size
    };

    set_object_name(
        instance,
        device,
        &format!("{} Buffer", name),
        buffer,
    )?;

    set_object_name(
        instance,
        device,
        &format!("{} Buffer Memory", name.to_owned()),
        buffer_memory,
    )?;

    Ok(wrapper)
}

/// # Safety
/// the `properties` field must contain flags that work together
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
pub fn copy_buffer_immediate(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    src: &Buffer,
    dst: &Buffer,
) -> Result<()> {
    unsafe {
        let command_buffer =
            begin_single_time_commands(instance, device, data, "Copy Buffer")?;

        let regions = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(src.size());
        device.cmd_copy_buffer(command_buffer, src.buffer(), dst.buffer(), &[regions]);

        end_single_time_commands(instance, device, data, command_buffer)?;
    }

    Ok(())
}

pub fn copy_buffer(device: &Device, command_buffer: vk::CommandBuffer, src: Buffer, dst: Buffer) {
    let region = vk::BufferCopy::default()
        .src_offset(0)
        .dst_offset(0)
        .size(src.size());

    unsafe { device.cmd_copy_buffer(command_buffer, src.buffer(), dst.buffer(), &[region]) };
}

/// Creates a [device local](vk::MemoryPropertyFlags::DEVICE_LOCAL) buffer and fills it with `contents`
pub fn create_and_stage_buffer<T>(
    instance: &Instance,
    device: &Device,
    data: &RendererData,
    usage: vk::BufferUsageFlags,
    name: &str,
    contents: &[T],
) -> Result<Buffer> {
    let size = size_of_val(contents) as u64;

    let mut staging_buffer = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        "",
    )?;

    staging_buffer.copy_vec_into_buffer(device, contents);

    let buffer = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        name,
    )?;

    copy_buffer_immediate(
        instance,
        device,
        data,
        &staging_buffer,
        &buffer,
    )?;

    staging_buffer.destroy(device);

    Ok(buffer)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl Buffer {
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn destroy(&mut self, device: &Device) {
        if !self.buffer.is_null() {
            unsafe {device.destroy_buffer(self.buffer, None);}
        }
        if !self.memory.is_null() {
            unsafe { device.free_memory(self.memory, None); }
        }
        self.buffer = vk::Buffer::null();
        self.memory = vk::DeviceMemory::null();
        self.size = 0;
    }

    /// Copies data into the buffer, if you are trying to copy a vec use [`copy_vec_into_buffer()`](Buffer::copy_vec_into_buffer)
    pub fn copy_data_into_buffer<T>(&self, device: &Device, data: &T) {
        let buffer_mem =
            unsafe { device.map_memory(self.memory, 0, size_of_val(data) as u64, vk::MemoryMapFlags::empty()) }
                .unwrap();

        unsafe { memcpy(data, buffer_mem.cast(), 1) };

        unsafe { device.unmap_memory(self.memory) };
    }

    /// Copies data into the buffer with an offset in the buffer, if you are trying to copy a vec use [`copy_vec_into_buffer_with_offset()`](Buffer::copy_vec_into_buffer_with_offset)
    pub fn copy_data_into_buffer_with_offset<T>(
        &self,
        device: &Device,
        data: &T,
        offset: u64,
    ) {
        let buffer_mem =
            unsafe { device.map_memory(self.memory, offset, size_of_val(data) as u64, vk::MemoryMapFlags::empty()) }
                .unwrap();

        unsafe { memcpy(data, buffer_mem.cast(), 1) };

        unsafe { device.unmap_memory(self.memory) };
    }

    /// Copies a vec into the buffer
    pub fn copy_vec_into_buffer<T>(&self, device: &Device, data: &[T]) {
        let buffer_mem =
            unsafe { device.map_memory(self.memory, 0, size_of_val(data) as u64, vk::MemoryMapFlags::empty()) }
                .unwrap();

        unsafe { memcpy(data.as_ptr(), buffer_mem.cast(), data.len()) };

        unsafe { device.unmap_memory(self.memory) };
    }

    /// Copies a vec into the buffer with an offset of `offset` in the buffer
    pub fn copy_vec_into_buffer_with_offset<T>(
        &self,
        device: &Device,
        data: &[T],
        offset: u64,
    ) {
        let buffer_mem =
            unsafe { device.map_memory(self.memory, offset, size_of_val(data) as u64, vk::MemoryMapFlags::empty()) }
                .unwrap();

        unsafe { memcpy(data.as_ptr(), buffer_mem.cast(), data.len()) };

        unsafe { device.unmap_memory(self.memory) };
    }

    pub fn copy_to_buffer(&self, device: &Device, command_buffer: vk::CommandBuffer, dst: &Buffer) {
        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(self.size);

        unsafe { device.cmd_copy_buffer(command_buffer, self.buffer, dst.buffer(), &[region]) };
    }

    pub fn copy_to_buffer_immediate(&self, instance: &Instance, device: &Device, data: &RendererData, dst: &Buffer) -> Result<()> {
        let command_buffer = unsafe { begin_single_time_commands(instance, device, data, "Copy Buffer") }?;
        
        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(self.size);

        unsafe { device.cmd_copy_buffer(command_buffer, self.buffer, dst.buffer(), &[region]) };

        unsafe { end_single_time_commands(instance, device, data, command_buffer) }?;

        Ok(())
    }
}
