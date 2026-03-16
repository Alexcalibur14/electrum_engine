use ash::vk;
use ash::Device;

pub fn bind_index_vertex(device: &Device, command_buffer: vk::CommandBuffer, index_buffer: vk::Buffer, index_type: vk::IndexType, vertex_buffer: vk::Buffer) {
    unsafe {
        device.cmd_bind_index_buffer(command_buffer, index_buffer, 0, index_type);
        device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
    }
}
