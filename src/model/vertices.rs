use std::hash::{Hash, Hasher};
use std::mem::size_of;

use glam::{Vec2, Vec3};
use vulkanalia::prelude::v1_2::*;

use crate::Vertex;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PCTVertex {
    pub pos: Vec3,
    pub tex_coord: Vec2,
    pub normal: Vec3,
}

impl Vertex for PCTVertex {
    fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<PCTVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(size_of::<Vec3>() as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec3>() as u32 + size_of::<Vec2>() as u32)
                .build(),
        ]
    }
}

impl PartialEq for PCTVertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.tex_coord == other.tex_coord && self.normal == other.normal
    }
}

impl Eq for PCTVertex {}

impl Hash for PCTVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
        self.normal[0].to_bits().hash(state);
        self.normal[1].to_bits().hash(state);
        self.normal[2].to_bits().hash(state);
    }
}
