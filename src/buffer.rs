use ash::vk;
use ash::{Device, Instance};
use anyhow::{Result, anyhow};

use std::ptr::copy_nonoverlapping as memcpy;

use crate::{begin_single_time_commands, end_single_time_commands, set_object_name, RendererData};

fn create_buffer(
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
        size,
        offset: 0,
        buffer_type: BufferType::default(),
        staging: None,
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
            .src_offset(src.offset())
            .dst_offset(dst.offset())
            .size(src.size());
        device.cmd_copy_buffer(command_buffer, src.buffer(), dst.buffer(), &[regions]);

        end_single_time_commands(instance, device, data, command_buffer)?;
    }

    Ok(())
}

pub fn copy_buffer(device: &Device, command_buffer: vk::CommandBuffer, src: Buffer, dst: Buffer) {
    let region = vk::BufferCopy::default()
        .src_offset(src.offset())
        .dst_offset(dst.offset())
        .size(src.size());

    unsafe { device.cmd_copy_buffer(command_buffer, src.buffer(), dst.buffer(), &[region]) };
}

/// Decides where the buffer will be allocated
#[derive(Debug, Clone, Copy, Default)]
pub enum BufferType {
    /// Allocates the buffer in Host memory (Main Memory) 
    #[default]
    HostLocal,
    /// Allocates the buffer in Device (GPU) memory
    DeviceLocal,
    /// Allocates a buffer in both Host and Device memory
    DeviceLocalStaged,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    offset: vk::DeviceSize,
    buffer_type: BufferType,
    staging: Option<(vk::Buffer, vk::DeviceMemory)>,
}

impl Buffer {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        buffer_type: BufferType,
        name: &str
    ) -> Result<Self> {
        match buffer_type {
            BufferType::HostLocal => {
                let mut buffer = create_buffer(instance, device, data, size, usage, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, name)?;
                buffer.buffer_type = buffer_type;
                Ok(buffer)
            },
            BufferType::DeviceLocal => {
                let mut buffer = create_buffer(instance, device, data, size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL, name)?;
                buffer.buffer_type = buffer_type;
                Ok(buffer)
            },
            BufferType::DeviceLocalStaged => {
                let mut buffer = create_buffer(instance, device, data, size, usage | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL, name)?;
                let staging = create_buffer(instance, device, data, size, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, name)?;

                buffer.staging = Some((staging.buffer, staging.memory));
                buffer.buffer_type = buffer_type;
                Ok(buffer)
            },
        }
    }

    pub fn create_and_load<T>(
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        usage: vk::BufferUsageFlags,
        buffer_type: BufferType,
        contents: &[T],
        name: &str
    ) -> Result<Buffer> {
        match buffer_type {
            BufferType::HostLocal => {
                let mut buffer = create_buffer(instance, device, data, size_of_val(contents) as vk::DeviceSize, usage, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, name)?;
                buffer.copy_array_into_buffer(instance, device, data, contents)?;
                buffer.buffer_type = buffer_type;

                Ok(buffer)
            },
            BufferType::DeviceLocal => {
                let mut buffer = create_buffer(instance, device, data, size_of_val(contents) as vk::DeviceSize, usage | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL, name)?;
                let mut staging = create_buffer(instance, device, data, size_of_val(contents) as vk::DeviceSize, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, name)?;
                staging.copy_array_into_buffer(instance, device, data, contents)?;
                staging.copy_to_buffer_immediate(instance, device, data, &buffer)?;

                staging.destroy(device);

                buffer.buffer_type = buffer_type;

                Ok(buffer)
            },
            BufferType::DeviceLocalStaged => {
                let mut buffer = create_buffer(instance, device, data, size_of_val(contents) as vk::DeviceSize, usage  | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL, name)?;
                let staging = create_buffer(instance, device, data, size_of_val(contents) as vk::DeviceSize,  vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, name)?;
                staging.copy_array_into_buffer(instance, device, data, contents)?;
                staging.copy_to_buffer_immediate(instance, device, data, &buffer)?;

                buffer.staging = Some((staging.buffer, staging.memory));
                buffer.buffer_type = buffer_type;
                Ok(buffer)
            },
        }
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    pub fn buffer_type(&self) -> BufferType {
        self.buffer_type
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe { device.destroy_buffer(self.buffer, None) };

        unsafe { device.free_memory(self.memory, None) };

        if let Some((staging_buffer, staging_memory)) = self.staging {
            unsafe { device.destroy_buffer(staging_buffer, None) };
            unsafe { device.free_memory(staging_memory, None) };

            self.staging = None;
        }

        self.buffer = vk::Buffer::null();
        self.memory = vk::DeviceMemory::null();
        self.size = 0;
        self.offset = 0;
        self.buffer_type = BufferType::default();
    }

    /// Copies data into the buffer, if you are trying to copy a vec use [`copy_array_into_buffer()`](Buffer::copy_array_into_buffer)
    pub fn copy_data_into_buffer<T>(&self, instance: &Instance, device: &Device, data: &RendererData, contents: &T) -> Result<()> {
        match self.buffer_type {
            BufferType::HostLocal => {
                let buffer_mem = unsafe {
                    device.map_memory(
                        self.memory,
                        self.offset,
                        size_of::<T>() as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents, buffer_mem.cast(), 1) };

                unsafe { device.unmap_memory(self.memory) };
            },
            BufferType::DeviceLocal => {
                let staging = create_buffer(
                    instance,
                    device,
                    data,
                    size_of::<T>() as vk::DeviceSize,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    "staging",
                )?;

                let staging_mem = unsafe {
                    device.map_memory(
                        staging.memory,
                        0,
                        size_of::<T>() as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents, staging_mem.cast(), 1) };

                unsafe { device.unmap_memory(staging.memory) };

                staging.copy_to_buffer_immediate(instance, device, data, self)?;
            },
            BufferType::DeviceLocalStaged => {
                // staging is guarateed to be some if the DeviceType is DeviceLocalStaged.
                let (_, staging_memory) = self.staging.unwrap();

                let staging_mem = unsafe {
                    device.map_memory(
                        staging_memory,
                        0,
                        size_of::<T>() as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents, staging_mem.cast(), 1) };

                unsafe { device.unmap_memory(staging_memory) };

                self.transfer_to_device_immediate(instance, device, data)?;
            },
        }

        Ok(())
    }

    /// Copies data into the buffer with an offset in the buffer, if you are trying to copy a vec use [`copy_array_into_buffer_with_offset()`](Buffer::copy_array_into_buffer_with_offset)
    pub fn copy_data_into_buffer_with_offset<T>(
        &self,
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        contents: &T,
        offset: vk::DeviceSize,
    ) -> Result<()> {
        match self.buffer_type {
            BufferType::HostLocal => {
                let buffer_mem = unsafe {
                    device.map_memory(
                        self.memory,
                        self.offset + offset,
                        size_of::<T>() as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents, buffer_mem.cast(), 1) };

                unsafe { device.unmap_memory(self.memory) };
            },
            BufferType::DeviceLocal => {
                let staging = create_buffer(
                    instance,
                    device,
                    data,
                    size_of::<T>() as vk::DeviceSize,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    "staging",
                )?;

                let staging_mem = unsafe {
                    device.map_memory(
                        staging.memory,
                        0,
                        size_of::<T>() as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents, staging_mem.cast(), 1) };

                unsafe { device.unmap_memory(staging.memory) };

                staging.copy_to_buffer_immediate_offset(instance, device, data, self, 0, offset)?;
            },
            BufferType::DeviceLocalStaged => {
                // staging is guarateed to be some if the DeviceType is DeviceLocalStaged.
                let (_, staging_memory) = self.staging.unwrap();

                let staging_mem = unsafe {
                    device.map_memory(
                        staging_memory,
                        offset,
                        size_of::<T>() as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents, staging_mem.cast(), 1) };

                unsafe { device.unmap_memory(staging_memory) };

                self.transfer_to_device_immediate_offset(instance, device, data, offset)?;
            },
        }

        Ok(())
    }

    /// Copies a vec into the buffer
    pub fn copy_array_into_buffer<T>(
        &self,
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        contents: &[T]
    ) -> Result<()> {
        match self.buffer_type {
            BufferType::HostLocal => {
                let buffer_mem = unsafe {
                    device.map_memory(
                        self.memory,
                        self.offset,
                        size_of_val(contents) as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents.as_ptr(), buffer_mem.cast(), contents.len()) };

                unsafe { device.unmap_memory(self.memory) };
            },
            BufferType::DeviceLocal => {
                let staging = create_buffer(
                    instance,
                    device,
                    data,
                    size_of_val(contents) as vk::DeviceSize,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    "staging",
                )?;

                let staging_mem = unsafe {
                    device.map_memory(
                        staging.memory,
                        0,
                        size_of_val(contents) as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents.as_ptr(), staging_mem.cast(), contents.len()) };

                unsafe { device.unmap_memory(staging.memory) };

                staging.copy_to_buffer_immediate(instance, device, data, self)?;
            },
            BufferType::DeviceLocalStaged => {
                // staging is guarateed to be some if the DeviceType is DeviceLocalStaged.
                let (_, staging_memory) = self.staging.unwrap();

                let staging_mem = unsafe {
                    device.map_memory(
                        staging_memory,
                        0,
                        size_of_val(contents) as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents.as_ptr(), staging_mem.cast(), contents.len()) };

                unsafe { device.unmap_memory(staging_memory) };

                self.transfer_to_device_immediate(instance, device, data)?;
            },
        }

        Ok(())
    }

    /// Copies a vec into the buffer with an offset of `offset` in the buffer
    pub fn copy_array_into_buffer_with_offset<T>(
        &self,
        instance: &Instance,
        device: &Device,
        data: &RendererData,
        contents: &[T],
        offset: vk::DeviceSize,
    ) -> Result<()> {
        match self.buffer_type {
            BufferType::HostLocal => {
                let buffer_mem = unsafe {
                    device.map_memory(
                        self.memory,
                        self.offset + offset,
                        size_of_val(contents) as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents.as_ptr(), buffer_mem.cast(), contents.len()) };

                unsafe { device.unmap_memory(self.memory) };
            },
            BufferType::DeviceLocal => {
                let staging = create_buffer(
                    instance,
                    device,
                    data,
                    size_of_val(contents) as vk::DeviceSize,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    "staging",
                )?;

                let staging_mem = unsafe {
                    device.map_memory(
                        staging.memory,
                        0,
                        size_of_val(contents) as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents.as_ptr(), staging_mem.cast(), contents.len()) };

                unsafe { device.unmap_memory(staging.memory) };

                staging.copy_to_buffer_immediate_offset(instance, device, data, self, 0, offset)?;
            },
            BufferType::DeviceLocalStaged => {
                // staging is guarateed to be some if the DeviceType is DeviceLocalStaged.
                let (_, staging_memory) = self.staging.unwrap();

                let staging_mem = unsafe {
                    device.map_memory(
                        staging_memory,
                        offset,
                        size_of_val(contents) as vk::DeviceSize,
                        vk::MemoryMapFlags::empty(),
                    )
                }?;

                unsafe { memcpy(contents.as_ptr(), staging_mem.cast(), contents.len()) };

                unsafe { device.unmap_memory(staging_memory) };

                self.transfer_to_device_immediate_offset(instance, device, data, offset)?;
            },
        }

        Ok(())
    }

    pub fn copy_to_buffer(&self, device: &Device, command_buffer: vk::CommandBuffer, dst: &Buffer) {
        let region = vk::BufferCopy::default()
            .src_offset(self.offset)
            .dst_offset(dst.offset)
            .size(self.size);

        unsafe { device.cmd_copy_buffer(command_buffer, self.buffer, dst.buffer(), &[region]) };
    }

    pub fn copy_to_buffer_offset(&self, device: &Device, command_buffer: vk::CommandBuffer, dst: &Buffer, src_offset: vk::DeviceSize, dst_offset: vk::DeviceSize, ) {
        let region = vk::BufferCopy::default()
            .src_offset(self.offset + src_offset)
            .dst_offset(dst.offset + dst_offset)
            .size(self.size - src_offset);

        unsafe { device.cmd_copy_buffer(command_buffer, self.buffer, dst.buffer(), &[region]) };
    }

    pub fn copy_to_buffer_immediate(&self, instance: &Instance, device: &Device, data: &RendererData, dst: &Buffer) -> Result<()> {
        let command_buffer = unsafe { begin_single_time_commands(instance, device, data, "Copy Buffer") }?;
        
        let region = vk::BufferCopy::default()
            .src_offset(self.offset)
            .dst_offset(dst.offset)
            .size(self.size);

        unsafe { device.cmd_copy_buffer(command_buffer, self.buffer, dst.buffer(), &[region]) };

        unsafe { end_single_time_commands(instance, device, data, command_buffer) }?;

        Ok(())
    }

    pub fn copy_to_buffer_immediate_offset(&self, instance: &Instance, device: &Device, data: &RendererData, dst: &Buffer, src_offset: vk::DeviceSize, dst_offset: vk::DeviceSize) -> Result<()> {
        let command_buffer = unsafe { begin_single_time_commands(instance, device, data, "Copy Buffer") }?;
        
        let region = vk::BufferCopy::default()
            .src_offset(self.offset + src_offset)
            .dst_offset(dst.offset + dst_offset)
            .size(self.size - src_offset);

        unsafe { device.cmd_copy_buffer(command_buffer, self.buffer, dst.buffer(), &[region]) };

        unsafe { end_single_time_commands(instance, device, data, command_buffer) }?;

        Ok(())
    }

    pub fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(self.offset)
            .range(self.size)
    }

    fn transfer_to_device_immediate(&self, instance: &Instance, device: &Device, data: &RendererData) -> Result<()> {
        match self.buffer_type {
            BufferType::HostLocal => panic!("transfer_to_device_immediate() is not supported for Host Local Buffers"),
            BufferType::DeviceLocal => panic!("transfer_to_device_immediate() is not supported for Device Local Buffers"),
            BufferType::DeviceLocalStaged => {},
        }

        let (staging_buffer, _) = self.staging.unwrap();

        let command_buffer =
            unsafe { begin_single_time_commands(instance, device, data, "Copy Buffer") }?;

        let regions = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(self.offset())
            .size(self.size());
        unsafe { device.cmd_copy_buffer(command_buffer, staging_buffer, self.buffer(), &[regions]) };

        unsafe { end_single_time_commands(instance, device, data, command_buffer) }?;
        
        Ok(())
    }

    fn transfer_to_device_immediate_offset(&self, instance: &Instance, device: &Device, data: &RendererData, offset: vk::DeviceSize) -> Result<()> {
        match self.buffer_type {
            BufferType::HostLocal => panic!("transfer_to_device_immediate() is not supported for Host Local Buffers"),
            BufferType::DeviceLocal => panic!("transfer_to_device_immediate() is not supported for Device Local Buffers"),
            BufferType::DeviceLocalStaged => {},
        }

        let (staging_buffer, _) = self.staging.unwrap();

        let command_buffer =
            unsafe { begin_single_time_commands(instance, device, data, "Copy Buffer") }?;

        let regions = vk::BufferCopy::default()
            .src_offset(offset)
            .dst_offset(self.offset() + offset)
            .size(self.size() - offset);
        unsafe { device.cmd_copy_buffer(command_buffer, staging_buffer, self.buffer(), &[regions]) };

        unsafe { end_single_time_commands(instance, device, data, command_buffer) }?;
        
        Ok(())
    }
}
