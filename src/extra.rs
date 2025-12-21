use std::f32::consts::PI;

use ash::{Device, Instance, vk};
use glam::{Mat4, Quat, Vec3, vec3, vec4};

use crate::{RendererData, descriptor::DescriptorBuilder, model::Object, resources::Handle};


#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraData {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
}

#[derive(Default, Clone)]
pub struct SimpleCamera {
    pub position: Vec3,
    pub rotation: Vec3,
    pub view: Mat4,
    pub projection: Projection,
    pub object_handle: Handle,
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
            Quat::from_euler(glam::EulerRot::XYZEx, rotation.x, rotation.y, rotation.z),
            position,
        );

        let mut camera_object = Object::new("main_camera");

        let camera_data = CameraData {
            view,
            proj: projection.proj,
            position,
        };

        camera_object.create_and_load_buffer_host(instance, device, data, &[camera_data], vk::BufferUsageFlags::UNIFORM_BUFFER, "camera_data");

        let camera_data_buffer = camera_object.buffers().items()[0];

        let buffer_info = [
            vk::DescriptorBufferInfo::default()
                .buffer(camera_data_buffer.buffer())
                .offset(0)
                .range(camera_data_buffer.size())
        ];

        let (descriptor_set, layout) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &buffer_info, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
            .build_data(device, data).unwrap();

        camera_object.add_descriptor_set(descriptor_set, "camera_data");
        data.layouts.push(layout, "main_camera");

        let object_handle = data.objects.push(camera_object, &["main_camera"]);

        SimpleCamera {
            position,
            rotation,
            view,
            projection,
            object_handle,
        }
    }

    pub fn new_view(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,
        view: Mat4,
        projection: Projection
    ) -> Self {
        let mut camera_object = Object::new("main_camera");

        let position = view.to_scale_rotation_translation().2;
        let rotation = view.to_scale_rotation_translation().1.to_euler(glam::EulerRot::XYZ);

        let camera_data = CameraData {
            view,
            proj: projection.proj,
            position,
        };

        camera_object.create_and_load_buffer_host(instance, device, data, &[camera_data], vk::BufferUsageFlags::UNIFORM_BUFFER, "camera_data");

        let camera_data_buffer = camera_object.buffers().items()[0];

        let buffer_info = [
            vk::DescriptorBufferInfo::default()
                .buffer(camera_data_buffer.buffer())
                .offset(0)
                .range(camera_data_buffer.size())
        ];

        let (descriptor_set, layout) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &buffer_info, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
            .build_data(device, data).unwrap();

        camera_object.add_descriptor_set(descriptor_set, "camera_data");
        data.layouts.push(layout, "main_camera");

        let object_handle = data.objects.push(camera_object, &["main_camera"]);

        SimpleCamera {
            position,
            rotation: vec3(rotation.0, rotation.1, rotation.2),
            view,
            projection,
            object_handle,
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

    pub fn rebuild(&mut self, device: &Device, data: &RendererData) {
        let camera_object = data.objects.get(&self.object_handle).unwrap();

        camera_object.buffers().get("camera_data").copy_data_into_buffer(
            device,
            &CameraData {
                view: self.view,
                proj: self.projection.proj,
                position: self.position,
            }
        );
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
        let correction = Mat4 { x_axis: vec4(1.0, 0.0, 0.0, 0.0), y_axis: vec4(0.0,-1.0, 0.0, 0.0), z_axis: vec4(0.0, 0.0, 0.5, 0.0), w_axis: vec4(0.0, 0.0, 0.5, 1.0) };
        let proj = correction * Mat4::perspective_rh(fov_y_rad, aspect_ratio, z_near, z_far);
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

pub fn radians(degrees: f32) -> f32 {
    (degrees / 180.0) * PI
}

pub fn degrees(radians: f32) -> f32 {
    (radians / PI) * 180.0
}

pub fn vec3_rad(x: f32, y: f32, z: f32) -> Vec3 {
    vec3(radians(x), radians(y), radians(z))
}

pub fn look_at(position: Vec3, target: Vec3) -> Vec3 {
    let direction = target - position;
    
    vec3(
        (direction.y).atan2(direction.z),
        (direction.x).atan2(direction.z),
        0.0,
    )
}
