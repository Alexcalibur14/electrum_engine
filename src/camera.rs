use glam::{Mat4, Quat, Vec3};

pub trait Camera {
    /// Returns the View matrix
    fn view(&self) -> Mat4;
    /// Returns the Projection matrix
    fn proj(&self) -> Mat4;
    /// Returns the Inverse Projection matrix
    fn inv_proj(&self) -> Mat4;

    /// calculate the view matrix
    fn calculate_view(&mut self);
    /// calculate the projection matrix
    fn calculate_proj(&mut self);
    /// calculate the sets the aspect for the projection matrix
    fn set_aspect(&mut self, aspect_ratio: f32);
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

    pub fn from_look_dir(&mut self, target: Vec3, up: Vec3) -> &mut Self {
        self.view = Mat4::look_at_rh(self.position, target, up);
        self
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

    fn calculate_view(&mut self) {
        self.view = Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, self.rotation.x, self.rotation.y, self.rotation.z), self.position);
    }

    fn calculate_proj(&mut self) {
        self.projection.recalculate();
    }

    fn set_aspect(&mut self, aspect_ratio: f32) {
        self.projection.aspect_ratio = aspect_ratio;
    }
}



#[derive(Debug, Clone, Default)]
pub struct Projection {
    pub fovy_rad: f32,
    pub aspect_ratio: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub proj: Mat4,
    pub inv_proj: Mat4,
}

impl Projection {
    pub fn new(fovy_rad: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let proj = Mat4::perspective_rh(fovy_rad, aspect_ratio, z_near, z_far);
        let inv_proj = proj.inverse();
        Projection {
            fovy_rad,
            aspect_ratio,
            z_near,
            z_far,
            proj,
            inv_proj
        }
    }

    pub fn recalculate(&mut self) -> &mut Self {
        let proj = Mat4::perspective_rh(self.fovy_rad, self.aspect_ratio, self.z_near, self.z_far);
        self.proj = proj;
        self.inv_proj = proj.inverse();
        self
    }
}
