use std::hash::{Hash, Hasher};

use electrum_engine_macros::Vertex;
use glam::{Vec2, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, Vertex)]
pub struct PCTVertex {
    #[vertex(format = 106)]
    pub pos: Vec3,
    #[vertex(format = 103)]
    pub tex_coord: Vec2,
    #[vertex(format = 106)]
    pub normal: Vec3,
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
