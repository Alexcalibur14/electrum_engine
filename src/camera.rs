use glam::{Mat4, Quat, Vec3};
use vulkanalia::prelude::v1_2::*;

use crate::buffer::BufferWrapper;

pub trait Camera {
    /// Returns the View matrix
    fn view(&self) -> Mat4;
    /// Returns the Projection matrix
    fn proj(&self) -> Mat4;
    /// Returns the Inverse Projection matrix
    fn inv_proj(&self) -> Mat4;

    /// Calculate the view matrix and update the view buffer with the new matrix
    fn calculate_view(&mut self, device: &Device, image_index: usize);

    /// Calculate the projection matrix and update the projection buffer with the new matrix
    fn calculate_proj(&mut self, device: &Device);
    /// Calculate the sets the aspect for the projection matrix
    fn set_aspect(&mut self, aspect_ratio: f32);

    fn view_buffer(&self, image_index: usize) -> BufferWrapper;
    fn proj_buffer(&self, image_index: usize) -> BufferWrapper;

    /// Used to destroy all buffers associated with the camera
    fn destroy(&self, device: &Device);

    /// Used to implement Clone on the trait
    fn clone_dyn(&self) -> Box<dyn Camera>;
}

impl Clone for Box<dyn Camera> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SimpleCamera {
    pub position: Vec3,
    pub rotation: Vec3,
    pub view: Mat4,
    pub projection: Projection,
}

impl SimpleCamera {
    pub fn new(position: Vec3, rotation: Vec3, projection: Projection) -> Self {
        let view = Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, rotation.x, rotation.y, rotation.z), position);

        SimpleCamera {
            position,
            rotation,
            view,
            projection,
        }
    }

    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        self.view = Mat4::look_at_rh(self.position, target, up);
        self.rotation = self.view.to_scale_rotation_translation().1.to_euler(glam::EulerRot::XYZ).into();
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

    fn calculate_view(&mut self, _device: &Device, _image_index: usize) {
        self.view = Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, self.rotation.x, self.rotation.y, self.rotation.z), self.position);
    }

    fn calculate_proj(&mut self, _device: &Device) {
        self.projection.recalculate();
    }

    fn set_aspect(&mut self, aspect_ratio: f32) {
        self.projection.aspect_ratio = aspect_ratio;
    }
    
    fn view_buffer(&self, _image_index: usize) -> BufferWrapper {
        BufferWrapper::default()
    }
    
    fn proj_buffer(&self, _image_index: usize) -> BufferWrapper {
        BufferWrapper::default()
    }
    
    fn destroy(&self, _device: &Device) {
    }
    
    fn clone_dyn(&self) -> Box<dyn Camera> {
        Box::new(self.clone())
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
            inv_proj
        }
    }

    pub fn recalculate(&mut self) -> &mut Self {
        let proj = Mat4::perspective_rh(self.fov_y_rad, self.aspect_ratio, self.z_near, self.z_far);

        self.proj = proj;
        self.inv_proj = proj.inverse();
        self
    }
}
