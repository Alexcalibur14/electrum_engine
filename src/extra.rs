use std::f32::consts::PI;

use ash::vk;
use glam::{Mat4, Quat, Vec3, vec2, vec3, vec4};

use crate::{RendererData, RenderingDevice, buffer::{Buffer, BufferType}, descriptor::DescriptorBuilder, model::{MeshData, OBJVertex, Object}, resources::Handle};


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
        device: &RenderingDevice,
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

        let camera_data_buffer = Buffer::create_and_load(device, data, vk::BufferUsageFlags::UNIFORM_BUFFER, BufferType::DeviceLocalStaged, &[camera_data], "camera_data").unwrap();

        let (descriptor_set, layout) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[camera_data_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
            .build_data(device, data).unwrap();

        camera_object.add_descriptor_set(descriptor_set, &[(camera_data_buffer, "camera_data")], &[], "camera_data");
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
        device: &RenderingDevice,
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

        let camera_data_buffer = Buffer::create_and_load(device, data, vk::BufferUsageFlags::UNIFORM_BUFFER, BufferType::DeviceLocalStaged, &[camera_data], "camera_data").unwrap();

        let (descriptor_set, layout) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[camera_data_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::ALL_GRAPHICS)
            .build_data(device, data).unwrap();

        camera_object.add_descriptor_set(descriptor_set, &[(camera_data_buffer, "camera_data")], &[], "camera_data");
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

    pub fn rebuild(&mut self, device: &RenderingDevice, data: &RendererData) {
        let camera_object = data.objects.get(&self.object_handle).unwrap();

        camera_object.buffers().get("camera_data").unwrap().copy_data_into_buffer(
            device,
            data,
            &CameraData {
                view: self.view,
                proj: self.projection.proj,
                position: self.position,
            }
        ).unwrap();
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
        let proj = perspective(fov_y_radians, aspect_ratio, z_near, z_far);
        
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
        let proj = perspective(self.fov_y_radians, self.aspect_ratio, self.z_near, self.z_far);

        self.proj = proj;
        self.inv_proj = proj.inverse();
    }
}


#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Light {
    data: LightData,
    object_handle: Handle,
}

impl Light {
    pub fn new(device: &RenderingDevice, data: &mut RendererData, light_data: LightData) -> Self {
        let light_buffer = Buffer::create_and_load(device, data, vk::BufferUsageFlags::UNIFORM_BUFFER, BufferType::DeviceLocal, &[light_data], "light_data").unwrap();
        
        let (light_descriptor, layout) = DescriptorBuilder::new()
            .bind_buffer(0, 1, &[light_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::FRAGMENT)
            .build_data(device, data).unwrap();

        let mut light_object = Object::new("light");
        light_object.add_descriptor_set(light_descriptor, &[(light_buffer, "light_data")], &[], "light_data");

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
    pub position: Vec3,
    pub strength: f32,
    pub direction: Vec3,
    pub light_type: LightType,
    pub colour: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum LightType {
    Point,
    #[default]
    Directional
}

pub fn create_debug_axes_object(device: &RenderingDevice, data: &mut RendererData, model: Mat4) -> Handle {
    let model_data = ModelData {
        model,
        normal: model.inverse().transpose(),
    };
    let object_buffer = Buffer::create_and_load(device, data, vk::BufferUsageFlags::UNIFORM_BUFFER, BufferType::DeviceLocal, &[model_data], "object_matrix").unwrap();

    let (object_descriptor, _) = DescriptorBuilder::new()
        .bind_buffer(0, 1, &[object_buffer.descriptor_info()], vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::VERTEX)
        .build_data(device, data).unwrap();

    let mut debug_axes = Object::new("axes");
    debug_axes.add_descriptor_set(object_descriptor, &[(object_buffer, "mvp")], &[], "mvp");

    *debug_axes.mesh_data_mut() = MeshData::new("axes");
    debug_axes.mesh_data_mut().build_vertex_staged(device, data, &[
        OBJVertex { position: vec3(0.0, 0.0, 0.0), normal: vec3(1.0, 0.0, 0.0), colour: vec3(1.0, 0.0, 0.0), uv: vec2(0.0, 0.0) },
        OBJVertex { position: vec3(1.0, 0.0, 0.0), normal: vec3(1.0, 0.0, 0.0), colour: vec3(1.0, 0.0, 0.0), uv: vec2(0.0, 0.0) },
        OBJVertex { position: vec3(0.0, 0.0, 0.0), normal: vec3(0.0, 1.0, 0.0), colour: vec3(0.0, 1.0, 0.0), uv: vec2(0.0, 0.0) },
        OBJVertex { position: vec3(0.0, 1.0, 0.0), normal: vec3(0.0, 1.0, 0.0), colour: vec3(0.0, 1.0, 0.0), uv: vec2(0.0, 0.0) },
        OBJVertex { position: vec3(0.0, 0.0, 0.0), normal: vec3(0.0, 0.0, 1.0), colour: vec3(0.0, 0.0, 1.0), uv: vec2(0.0, 0.0) },
        OBJVertex { position: vec3(0.0, 0.0, 1.0), normal: vec3(0.0, 0.0, 1.0), colour: vec3(0.0, 0.0, 1.0), uv: vec2(0.0, 0.0) },
    ]);

    data.objects.push(debug_axes, &["debug"])
}

/// Calculates a perspective projection matrix
/// 
/// taken from https://www.vincentparizet.com/blog/posts/vulkan_perspective_matrix/#implementation
pub fn perspective(fov_y_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let focal_length = 1.0 / (fov_y_radians / 2.0).tan();

    let a = focal_length / aspect;
    let b = -focal_length;
    let c = z_near / (z_far - z_near);
    let tz = z_far * c;

    Mat4 {
        x_axis: vec4(  a, 0.0, 0.0,  0.0),
        y_axis: vec4(0.0,   b, 0.0,  0.0),
        z_axis: vec4(0.0, 0.0,   c, -1.0),
        w_axis: vec4(0.0, 0.0,  tz,  0.0),
    }
}

/// Calculates an orthographic projection matrix
pub fn orthographic(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> Mat4 {
    let a  =            2.0  / (right - left);
    let b  =           -2.0  / (top - bottom);
    let c  =            1.0  / (far - near);
    let tx = -(right + left) / (right - left);
    let ty =  (top + bottom) / (top - bottom);
    let tz =            far  / (far - near);

    Mat4 {
        x_axis: vec4(  a, 0.0, 0.0, 0.0),
        y_axis: vec4(0.0,   b, 0.0, 0.0),
        z_axis: vec4(0.0, 0.0,   c, 0.0),
        w_axis: vec4( tx,  ty,  tz, 1.0),
    }
}

pub fn orthographic_symetric(width: f32, height: f32, near: f32, far: f32) -> Mat4 {
    let a  =  2.0 / width;
    let b  = -2.0 / height;
    let c  =  1.0 / (far - near);
    let tz =  far / (far - near);

    Mat4 {
        x_axis: vec4(  a, 0.0, 0.0,  0.0),
        y_axis: vec4(0.0,   b, 0.0,  0.0),
        z_axis: vec4(0.0, 0.0,   c,  0.0),
        w_axis: vec4(0.0, 0.0,  tz,  1.0),
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

pub fn to_ndc(vec: Vec3) -> Vec3 {
    vec3(vec.x, -vec.y, vec.z)
}

pub fn look_at(position: Vec3, target: Vec3) -> Vec3 {
    let direction = (target - position).normalize();
    
    vec3(
        direction.y.atan2(direction.z),
        direction.x.atan2(direction.z),
        0.0,
    )
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[repr(C)]
pub struct ModelData {
    pub model: Mat4,
    pub normal: Mat4,
}

impl ModelData {
    pub fn new(model_matrix: Mat4) -> ModelData {
        ModelData {
            model: model_matrix,
            normal: model_matrix.inverse().transpose(),
        }
    }
}
