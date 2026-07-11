use std::{collections::HashMap, hash::{DefaultHasher, Hash, Hasher}, marker::PhantomData};

use thiserror::Error;

use ash::vk;
use tracing::info;
use anyhow::Result;

use crate::{RendererData, RenderingDevice};

const STANDARD_SIZES: [(vk::DescriptorType, u32); 10] = [
    (vk::DescriptorType::SAMPLER, 1),
    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 50),
    (vk::DescriptorType::SAMPLED_IMAGE, 1),
    (vk::DescriptorType::STORAGE_IMAGE, 1),
    (vk::DescriptorType::UNIFORM_TEXEL_BUFFER, 1),
    (vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1),
    (vk::DescriptorType::UNIFORM_BUFFER, 50),
    (vk::DescriptorType::STORAGE_BUFFER, 50),
    (vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, 1),
    (vk::DescriptorType::STORAGE_BUFFER_DYNAMIC, 1),
];

#[derive(Debug, Clone)]
pub struct DescriptorAllocator {
    sizes: Vec<(vk::DescriptorType, u32)>,
    count: u32,
    current_pool: Option<vk::DescriptorPool>,
    used_pools: Vec<vk::DescriptorPool>,
    free_pools: Vec<vk::DescriptorPool>,
    descriptor_layouts: Vec<Vec<(vk::DescriptorSetLayout, vk::DescriptorSet, bool)>>,
}

impl DescriptorAllocator {
    pub fn new() -> Self {
        DescriptorAllocator {
            sizes: STANDARD_SIZES.to_vec(),
            count: 100,
            current_pool: None,
            used_pools: vec![],
            free_pools: vec![],
            descriptor_layouts: vec![],
        }
    }

    pub fn new_custom(sizes: Vec<(vk::DescriptorType, u32)>, count: u32) -> Self {
        DescriptorAllocator {
            sizes,
            count,
            current_pool: None,
            used_pools: vec![],
            free_pools: vec![],
            descriptor_layouts: vec![]
        }
    }

    fn get_pool(&mut self, device: &RenderingDevice) -> Result<vk::DescriptorPool, DescriptorAllocateError> {
        if self.free_pools.len() > 0 {
            Ok(self.free_pools.pop().unwrap())
        }
        else {
            self.create_pool(device)
        }
    }

    fn create_pool(&self, device: &RenderingDevice) -> Result<vk::DescriptorPool, DescriptorAllocateError> {
        let sizes = self.sizes.iter().map(|(d_type, amt)| {
            vk::DescriptorPoolSize::default()
                .ty(*d_type)
                .descriptor_count(amt * self.count)
        }).collect::<Vec<_>>();
        
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .pool_sizes(&sizes)
            .max_sets(self.count);

        match unsafe { device.create_descriptor_pool(&create_info, None) } {
            Ok(pool) => Ok(pool),
            Err(_) => Err(DescriptorAllocateError::UnableToCreate),
        }
    }

    pub fn allocate(&mut self, device: &RenderingDevice, set_layout: &vk::DescriptorSetLayout) -> Result<vk::DescriptorSet, DescriptorAllocateError> {
        for pool in self.descriptor_layouts.iter_mut() {
            for (layout, set, allocated) in pool.iter_mut() {
                if *allocated {
                    continue;
                }

                if layout != set_layout {
                    continue;
                }

                *allocated = true;
                return Ok(*set);
            }
        }
        
        if self.current_pool.is_none() {
            let pool = self.get_pool(device).unwrap();
            self.current_pool = Some(pool);
            self.used_pools.push(pool);
            self.descriptor_layouts.push(vec![]);
        }

        let pool = self.current_pool.unwrap();

        let layouts = vec![*set_layout];
        let mut allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let result = unsafe { device.allocate_descriptor_sets(&allocate_info) };

        match result {
            Ok(sets) => {
                self.descriptor_layouts.last_mut().unwrap().push((*set_layout, sets[0], true));
                return Ok(sets[0])
            },
            Err(err) => match err {
                vk::Result::ERROR_FRAGMENTED_POOL => {},
                vk::Result::ERROR_OUT_OF_POOL_MEMORY => {},

                _ => return Err(DescriptorAllocateError::AllocationFailed(err)),
            },
        }

        let pool = self.get_pool(device).unwrap();
        self.current_pool = Some(pool);
        self.used_pools.push(pool);
        self.descriptor_layouts.push(vec![]);

        allocate_info.descriptor_pool = pool;

        let result = unsafe { device.allocate_descriptor_sets(&allocate_info) };

        match result {
            Ok(sets) => {
                self.descriptor_layouts.last_mut().unwrap().push((*set_layout, sets[0], true));
                return Ok(sets[0])
            },
            Err(err) => {
                return Err(DescriptorAllocateError::AllocationFailed(err))
            },
        }
    }

    pub fn reset_pools(&mut self, device: &RenderingDevice) {
        for pool in self.used_pools.clone() {
            unsafe { device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty()) }.unwrap();
            self.free_pools.push(pool);
        }
        self.used_pools.clear();
        self.descriptor_layouts.clear();
    }

    pub fn deallocate_descriptor(&mut self, device: &RenderingDevice, descriptor_set: vk::DescriptorSet) -> Result<()> {
        let mut freed_pools = vec![];
        
        for (id, pool) in self.descriptor_layouts.iter_mut().enumerate() {
            let total = pool.len();
            let mut freed: usize = 0;
            
            for (_, set, allocated) in pool.iter_mut() {
                match allocated {
                    true => {
                        if *set != descriptor_set {
                            continue;
                        }
                    },
                    false => {
                        freed += 1;
                        continue
                    },
                }

                *allocated = false;
                freed += 1;
            }

            if freed == total {
                pool.clear();
                freed_pools.push(id);
            }
        }

        for id in freed_pools {
            self.descriptor_layouts.remove(id);
            let pool = self.used_pools.remove(id);

            unsafe { device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty()) }?;
            self.free_pools.push(pool);
        }

        self.current_pool = self.used_pools.last().cloned();
        
        Ok(())
    }

    pub fn destroy(&self, device: &RenderingDevice) {
        for pool in &self.used_pools {
            unsafe { device.destroy_descriptor_pool(*pool, None) };
        }
        for pool in &self.free_pools {
            unsafe { device.destroy_descriptor_pool(*pool, None) };
        }
    }

    pub fn print_allocations(&self) {
        for (id, pool) in self.descriptor_layouts.iter().enumerate() {
            info!("pool: {}; {} allocations", id, pool.len());
            let mut string = String::new();
            for (layout, _, allocated) in pool {
                match allocated {
                    true => string.push_str(&format!("\x1B[32m{:?}\x1B[0m,", layout)),
                    false => string.push_str(&format!("\x1B[31m{:?}\x1B[0m,", layout)),
                }
            }
            info!("layouts: {}", string);
        }
    }
}

impl Default for DescriptorAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Error)]
pub enum DescriptorAllocateError {
    #[error("Unable to create another pool")]
    UnableToCreate,
    #[error("Descriptor set allocation failed due to error: {0}")]
    AllocationFailed(vk::Result),
}


#[derive(Debug, Clone)]
pub struct DescriptorLayoutCache {
    cache: HashMap<u64, vk::DescriptorSetLayout>,
}

impl DescriptorLayoutCache {
    pub fn new() -> Self {
        DescriptorLayoutCache {
            cache: HashMap::new(),
        }
    }

    pub fn create_descriptor_set_layout(&mut self, device: &RenderingDevice, bindings: &[vk::DescriptorSetLayoutBinding]) -> vk::DescriptorSetLayout {
        let hash = Self::hash_bindings(bindings);

        match self.cache.get(&hash) {
            Some(layout) => *layout,
            None => {
                let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(bindings);

                let layout = unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap();
                self.cache.insert(hash, layout.clone());
                layout
            },
        }
    }

    fn hash_bindings(bindings: &[vk::DescriptorSetLayoutBinding]) -> u64 {
        let mut hasher = DefaultHasher::new();
        bindings.iter().for_each(|b| {
            b.binding.hash(&mut hasher);
            b.descriptor_type.hash(&mut hasher);
            b.descriptor_count.hash(&mut hasher);
            b.stage_flags.hash(&mut hasher);
            b.p_immutable_samplers.hash(&mut hasher);
        });
        hasher.finish()
    }

    pub fn destroy(&mut self, device: &RenderingDevice) {
        self.cache.iter().for_each(|(_, v)| unsafe { device.destroy_descriptor_set_layout(*v, None) });
    }
}

impl Default for DescriptorLayoutCache {
    fn default() -> Self {
        Self::new()
    }
}


#[derive(Debug, Clone)]
pub struct DescriptorBuilder<'a> {
    writes: Vec<vk::WriteDescriptorSet<'a>>,
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
    phantom: PhantomData<&'a u8>,
}

impl<'a> DescriptorBuilder<'a> {
    pub fn new() -> Self {
        DescriptorBuilder {
            writes: vec![],
            bindings: vec![],
            phantom: PhantomData,
        }
    }

    pub fn bind_buffer(&'a mut self, binding: u32, count: u32, buffer_info: &'a[vk::DescriptorBufferInfo; 1], descriptor_type: vk::DescriptorType, stage_flags: vk::ShaderStageFlags) -> &'a mut Self {
        let new_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(count)
            .stage_flags(stage_flags);

        self.bindings.push(new_binding);

        let write_info = vk::WriteDescriptorSet::default()
            .buffer_info(buffer_info)
            .descriptor_type(descriptor_type)
            .dst_binding(binding);

        self.writes.push(write_info);

        self
    }

    pub fn bind_image(&'a mut self, binding: u32, count: u32, image_info: &'a[vk::DescriptorImageInfo; 1], descriptor_type: vk::DescriptorType, stage_flags: vk::ShaderStageFlags) -> &'a mut Self {
        let new_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(count)
            .stage_flags(stage_flags);

        self.bindings.push(new_binding);

        let write_info = vk::WriteDescriptorSet::default()
            .image_info(image_info)
            .descriptor_type(descriptor_type)
            .dst_binding(binding);

        self.writes.push(write_info);

        self
    }

    pub fn build(&'a mut self, device: &RenderingDevice, layout_cache: &mut DescriptorLayoutCache, allocator: &mut DescriptorAllocator) -> Result<(vk::DescriptorSet, vk::DescriptorSetLayout), DescriptorAllocateError> {
        let layout = layout_cache.create_descriptor_set_layout(device, &self.bindings);
        let set = allocator.allocate(device, &layout)?;

        for write in self.writes.iter_mut() {
            write.dst_set = set;
        }

        unsafe { device.update_descriptor_sets(&self.writes, &[]) };

        Ok((set, layout))
    }

    pub fn build_data(&'a mut self, device: &RenderingDevice, data: &mut RendererData) -> Result<(vk::DescriptorSet, vk::DescriptorSetLayout), DescriptorAllocateError> {
        let layout = data.descriptor_layout_cache.create_descriptor_set_layout(device, &self.bindings);
        let set = data.descriptor_pool.allocate(device, &layout)?;

        for write in self.writes.iter_mut() {
            write.dst_set = set;
        }

        unsafe { device.update_descriptor_sets(&self.writes, &[]) };

        Ok((set, layout))
    }
}

impl<'a> Default for DescriptorBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn update_image_binding(device: &RenderingDevice, image_info: &[vk::DescriptorImageInfo; 1], set: vk::DescriptorSet, binding: u32, descriptor_type: vk::DescriptorType) {
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(descriptor_type)
        .image_info(image_info);

    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

pub fn update_buffer_binding(device: &RenderingDevice, buffer_info: &[vk::DescriptorBufferInfo; 1], set: vk::DescriptorSet, binding: u32, descriptor_type: vk::DescriptorType) {
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(descriptor_type)
        .buffer_info(buffer_info);

    unsafe { device.update_descriptor_sets(&[write], &[]) };
}
