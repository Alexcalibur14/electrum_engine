use std::f32::consts::PI;

use ash::{Device, Instance, vk};
use glam::{vec2, vec3};

use crate::{RendererData, extra::lerp, model::{MeshData, OBJVertex, Object}, resources::Handle};

pub struct Plane {
    width: f32,
    height: f32,
    object_handle: Handle,
}

impl Plane {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,
        width: f32,
        height: f32,
        x_subdivisions: u32,
        z_subdivisions: u32,
        tags: &[&'static str]
    ) -> Self {
        let neg_x = -width / 2.0;
        let pos_x =  width / 2.0;

        let neg_z = -height / 2.0;
        let pos_z =  height / 2.0;

        let mut vertices = vec![];
        let mut indices = vec![];

        for z_i in 0..=z_subdivisions {
            let z_percent = lerp(0.0, 1.0, z_i as f32 / z_subdivisions as f32);
            let z = lerp(neg_z, pos_z, z_percent);

            for x_i in 0..=x_subdivisions {
                let x_percent = lerp(0.0, 1.0, x_i as f32 / x_subdivisions as f32);
                let x = lerp(neg_x, pos_x, x_percent);

                vertices.push(OBJVertex { position: vec3(x, 0.0, z), normal: vec3(0.0, 1.0, 0.0), colour: vec3(0.0, 0.0, 0.0), uv: vec2(x_percent, z_percent) });

                if z_i == z_subdivisions || x_i == x_subdivisions {
                    continue;
                }

                indices.push( z_i      * (z_subdivisions + 1) + x_i);
                indices.push((z_i + 1) * (z_subdivisions + 1) + x_i);
                indices.push( z_i      * (z_subdivisions + 1) + x_i + 1);

                indices.push( z_i      * (z_subdivisions + 1) + x_i + 1);
                indices.push((z_i + 1) * (z_subdivisions + 1) + x_i);
                indices.push((z_i + 1) * (z_subdivisions + 1) + x_i + 1);
            }
        }
        

        let mut mesh_data = MeshData::new("Plane");
        mesh_data.build_vertex_staged(instance, device, data, &vertices);
        mesh_data.build_index_staged(instance, device, data, &indices, vk::IndexType::UINT32);

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

pub struct UVSphere {
    radius: f32,
    object_handle: Handle,
}

impl UVSphere {
    pub fn new(
        instance: &Instance,
        device: &Device,
        data: &mut RendererData,
        radius: f32,
        h_subdivisions: u32,
        v_subdivisions: u32,
        tags: &[&'static str]
    ) -> UVSphere {
        let mut vertices = vec![];
        let mut indices = vec![];

        let v_subdivisions = 2 * v_subdivisions;
        let h_subdivisions = 4 * h_subdivisions;

        for x_i in 0..=h_subdivisions {
            let x_percent = lerp(0.0, 1.0, x_i as f32 / h_subdivisions as f32);
            
            let (x_norm, z_norm) = (2.0 * PI * x_percent).sin_cos();
            
            for y_i in 0..=v_subdivisions {
                let y_percent = lerp(1.0, 0.0, y_i as f32 / v_subdivisions as f32);
                
                let (h_radius, y_norm) = (PI * y_percent).sin_cos();

                vertices.push(
                    OBJVertex {
                        position: vec3(
                            x_norm * h_radius * radius,
                            -y_norm * radius,
                            z_norm * h_radius * radius,
                        ),
                        normal: vec3(
                            x_norm * h_radius,
                            -y_norm,
                            z_norm * h_radius
                        ),
                        colour: vec3(
                            0.0,
                            0.0,
                            0.0
                        ),
                        uv: vec2(
                            x_percent,
                            1.0 - y_percent
                        ),
                    }
                );

                if x_i == h_subdivisions || y_i == v_subdivisions {
                    continue;
                }

                indices.push( x_i      * (v_subdivisions + 1) + y_i);
                indices.push( x_i      * (v_subdivisions + 1) + y_i + 1);
                indices.push((x_i + 1) * (v_subdivisions + 1) + y_i);

                indices.push( x_i      * (v_subdivisions + 1) + y_i + 1);
                indices.push((x_i + 1) * (v_subdivisions + 1) + y_i + 1);
                indices.push((x_i + 1) * (v_subdivisions + 1) + y_i);
            }
        }

        let mut mesh_data = MeshData::new("Plane");
        mesh_data.build_vertex_staged(instance, device, data, &vertices);
        mesh_data.build_index_staged(instance, device, data, &indices, vk::IndexType::UINT32);

        let mut plane_object = Object::new("Plane");
        *plane_object.mesh_data_mut() = mesh_data;

        let handle = data.objects.push(plane_object, tags);

        UVSphere {
            radius,
            object_handle: handle,
        }
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn object(&self) -> Handle {
        self.object_handle
    }
}
