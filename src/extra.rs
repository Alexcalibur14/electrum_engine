use std::f32::consts::PI;

use ash::{Device, Instance, vk};
use glam::{Mat4, Quat, Vec3, vec3, vec4};

use crate::{RendererData, buffer::Buffer, descriptor::DescriptorBuilder, model::{MeshData, OBJVertex, Object}, resources::Handle};


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
            .bind_buffer(0, 1, &buffer_info, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::ALL_GRAPHICS)
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
    pub fov_y_radians: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub proj: Mat4,
    pub inv_proj: Mat4,
}

impl Projection {
    pub fn new(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let proj = projection(fov_y_radians, aspect_ratio, z_near, z_far);
        
        let inv_proj = proj.inverse();

        Projection {
            fov_y_radians,
            aspect_ratio,
            z_near,
            z_far,
            proj,
            inv_proj,
        }
    }

    pub fn recalculate(&mut self) {
        let proj = projection(self.fov_y_radians, self.aspect_ratio, self.z_near, self.z_far);

        self.proj = proj;
        self.inv_proj = proj.inverse();
    }
}

pub struct Plane {
    width: f32,
    height: f32,
    object_handle: Handle,
}

impl Plane {
    pub fn new(instance: &Instance, device: &Device, data: &mut RendererData, width: f32, height: f32, tags: &[&'static str]) -> Self {
        let vertices = [
            OBJVertex { position: [-width / 2.0, 0.0, -height / 2.0], normal: [0.0, 1.0, 0.0], colour: [0.0, 0.0, 0.0], uv: [0.0, 0.0] },
            OBJVertex { position: [ width / 2.0, 0.0, -height / 2.0], normal: [0.0, 1.0, 0.0], colour: [0.0, 0.0, 0.0], uv: [1.0, 0.0] },
            OBJVertex { position: [-width / 2.0, 0.0,  height / 2.0], normal: [0.0, 1.0, 0.0], colour: [0.0, 0.0, 0.0], uv: [0.0, 1.0] },
            OBJVertex { position: [ width / 2.0, 0.0,  height / 2.0], normal: [0.0, 1.0, 0.0], colour: [0.0, 0.0, 0.0], uv: [1.0, 1.0] },
        ];

        let indices = [
            0, 2, 1,
            1, 2, 3u16
        ];

        let mut mesh_data = MeshData::new("Plane");
        mesh_data.build_vertex_staged(instance, device, data, &vertices);
        mesh_data.build_index_staged(instance, device, data, &indices, vk::IndexType::UINT16);

        let mut plane_object = Object::new("Plane");
        *plane_object.mesh_data_mut() = mesh_data;

        let handle = data.objects.push(plane_object, tags);
        
        Plane {
            width,
            height,
            object_handle: handle,
        }
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn height(&self) -> f32 {
        self.height
    }

    pub fn object(&self) -> Handle {
        self.object_handle
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Light {
    data: LightData,
    object_handle: Handle,
}

impl Light {
    pub fn new(instance: &Instance, device: &Device, data: &mut RendererData, light_data: LightData) -> Self {
        let light_buffer = Buffer::create_and_stage(instance, device, data, &[light_data], vk::BufferUsageFlags::UNIFORM_BUFFER, "light_data");
        
        let buffer_info = &[
            vk::DescriptorBufferInfo::default()
                .buffer(light_buffer.buffer())
                .offset(0)
                .range(light_buffer.size())
        ];

        let (light_descriptor, layout) = DescriptorBuilder::new()
            .bind_buffer(0, 1, buffer_info, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::FRAGMENT)
            .build_data(device, data).unwrap();

        let mut light_object = Object::new("light");
        light_object.add_buffer(light_buffer, "light_data");
        light_object.add_descriptor_set(light_descriptor, "light_data");

        data.layouts.push(layout, "light_data");
        let object_handle = data.objects.push(light_object, &["light"]);

        Light {
            data: light_data,
            object_handle,
        }
    }

    pub fn data(&self) -> &LightData {
        &self.data
    }

    pub fn object(&self) -> &Handle {
        &self.object_handle
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LightData {
    pub position: [f32; 3],
    pub strength: f32,
    pub direction: [f32; 3],
    pub light_type: LightType,
    pub colour: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum LightType {
    Point,
    #[default]
    Directional
}

/// Calculates a perspective projection matrix
/// 
/// taken from https://www.vincentparizet.com/blog/posts/vulkan_perspective_matrix/#implementation
pub fn projection(fov_y_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let focal_length = 1.0 / (fov_y_radians / 2.0).tan();

    let x = focal_length / aspect;
    let y = -focal_length;
    let a = z_near / (z_far - z_near);
    let b = z_far * a;

    Mat4 {
        x_axis: vec4(x, 0.0, 0.0, 0.0),
        y_axis: vec4(0.0,   y, 0.0,  0.0),
        z_axis: vec4(0.0, 0.0,   a, -1.0),
        w_axis: vec4(0.0, 0.0,   b,  0.0),
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
