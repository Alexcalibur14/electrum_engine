use std::mem::size_of;

use glam::{Mat4, Quat, Vec3};

use ash::vk;
use ash::{Device, Instance};

use crate::{
    buffer::{create_buffer, BufferWrapper}, DescriptorBuilder, Loadable, RendererData
};

pub trait Camera {
    /// Returns the View matrix
    fn view(&self) -> Mat4;
    /// Returns the Projection matrix
    fn proj(&self) -> Mat4;
    /// Returns the Inverse Projection matrix
    fn inv_proj(&self) -> Mat4;

    fn get_data(&self) -> CameraData;
    fn get_descriptor_sets(&self) -> Vec<vk::DescriptorSet>;
    fn get_set_layout(&self) -> vk::DescriptorSetLayout;

    /// Calculate the view matrix and update the view buffer with the new matrix
    fn calculate_view(&mut self, device: &Device, image_index: usize);

    /// Calculate the projection matrix and update the projection buffer with the new matrix
    fn calculate_proj(&mut self, device: &Device);
    /// Calculate the sets the aspect for the projection matrix
    fn set_aspect(&mut self, aspect_ratio: f32);

    /// Used to destroy all buffers associated with the camera
    fn destroy(&self, device: &Device);

    /// Used to implement Clone on the trait
    fn clone_dyn(&self) -> Box<dyn Camera>;

    fn loaded(&self) -> bool;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraData {
    pub position: Vec3,
    pub _padd: u32,
    pub view: Mat4,
    pub proj: Mat4,
}

impl Clone for Box<dyn Camera> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

impl Loadable for Box<dyn Camera> {
    fn is_loaded(&self) -> bool {
        self.loaded()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SimpleCamera {
    pub position: Vec3,
    pub rotation: Vec3,
    pub view: Mat4,
    pub projection: Projection,

    descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub buffers: Vec<BufferWrapper>,

    loaded: bool,
}

impl SimpleCamera {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,
        position: Vec3,
        rotation: Vec3,
        projection: Projection,
    ) -> Self {
        let view = Mat4::from_rotation_translation(
            Quat::from_euler(glam::EulerRot::XYZ, rotation.x, rotation.y, rotation.z),
            position,
        );

        let camera_data = CameraData {
            position,
            _padd: 0,
            view,
            proj: Mat4::default(),
        };

        let mut buffers = vec![];

        for i in 0..data.swapchain_images.len() {
            let buffer = create_buffer(
                instance,
                device,
                data,
                size_of::<CameraData>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                &format!("Camera Data {}", i),
            )
            .unwrap();

            buffer.copy_data_into_buffer(device, &camera_data);

            buffers.push(buffer);
        }

        let mut descriptor_sets = vec![];
        let mut descriptor_set_layout = Default::default();

        for i in 0..data.swapchain_images.len() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers[i].buffer)
                .offset(0)
                .range(size_of::<CameraData>() as u64);

            let (descriptor_set, new_set_layout) = DescriptorBuilder::new()
                .bind_buffer(0, 1, &[buffer_info], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
                .build(device, &mut data.global_layout_cache, &mut data.global_descriptor_pools).unwrap();

            descriptor_sets.push(descriptor_set);
            descriptor_set_layout = new_set_layout;
        }

        SimpleCamera {
            position,
            rotation,
            view,
            projection,

            descriptor_set_layout,
            descriptor_sets,
            buffers,

            loaded: true,
        }
    }

    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        self.view = Mat4::look_at_rh(self.position, target, up);
        self.rotation = self
            .view
            .to_scale_rotation_translation()
            .1
            .to_euler(glam::EulerRot::XYZ)
            .into();
        self.position = self.view.to_scale_rotation_translation().2;
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

    fn calculate_view(&mut self, device: &Device, image_index: usize) {
        self.view = Mat4::from_rotation_translation(
            Quat::from_euler(
                glam::EulerRot::XYZ,
                self.rotation.x,
                self.rotation.y,
                self.rotation.z,
            ),
            self.position,
        );

        let camera_data = CameraData {
            position: self.position,
            _padd: 0,
            view: self.view,
            proj: self.proj(),
        };

        let buffer = self.buffers[image_index];

        buffer.copy_data_into_buffer(device, &camera_data);
    }

    fn calculate_proj(&mut self, device: &Device) {
        self.projection.recalculate();

        let camera_data = CameraData {
            position: self.position,
            _padd: 0,
            view: self.view,
            proj: self.proj(),
        };

        for buffer in self.buffers.clone() {
            buffer.copy_data_into_buffer(device, &camera_data);
        }
    }

    fn set_aspect(&mut self, aspect_ratio: f32) {
        self.projection.aspect_ratio = aspect_ratio;
    }

    fn destroy(&self, device: &Device) {
        self.buffers.iter().for_each(|b| b.destroy(device));
    }

    fn clone_dyn(&self) -> Box<dyn Camera> {
        Box::new(self.clone())
    }

    fn get_data(&self) -> CameraData {
        CameraData {
            position: self.position,
            _padd: 0,
            view: self.view,
            proj: self.projection.proj,
        }
    }

    fn get_descriptor_sets(&self) -> Vec<vk::DescriptorSet> {
        self.descriptor_sets.clone()
    }

    fn get_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
    
    fn loaded(&self) -> bool {
        self.loaded
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
            inv_proj,
        }
    }

    pub fn recalculate(&mut self) {
        let proj = Mat4::perspective_rh(self.fov_y_rad, self.aspect_ratio, self.z_near, self.z_far);

        self.proj = proj;
        self.inv_proj = proj.inverse();
    }
}
