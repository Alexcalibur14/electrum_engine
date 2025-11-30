use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle<T> {
    handle: usize,
    _phantom: PhantomData<T>,
}

impl<T> Handle<T> {
    pub fn new(handle: usize) -> Self {
        Handle {
            handle,
            _phantom: PhantomData,
        }
    }

    pub fn index(&self) -> usize {
        self.handle
    }
}

pub struct Collection<'a, T> {
    items: Vec<T>,
    tags: Vec<Vec<&'a str>>
}

impl<'a, T> Collection<'a, T> {
    pub fn new() -> Self {
        Collection {
            items: vec![],
            tags: vec![]
        }
    }

    pub fn push(&mut self, item: T, tags: &[&'a str]) -> Handle<T> {
        let index = self.items.len();
        self.items.push(item);
        self.tags.push(tags.to_vec());

        Handle::new(index)
    }

    pub fn get(&self, handle: &Handle<T>) -> Option<&T> {
        self.items.get(handle.index())
    }

    pub fn get_mut(&mut self, handle: &Handle<T>) -> Option<&mut T> {
        self.items.get_mut(handle.index())
    }

    pub fn get_with_tag(&self, tag: &str) -> Vec<&T> {
        self.tags.iter().enumerate().filter_map(|(index, tags)| {
            if tags.contains(&tag) {
                Some(self.items.get(index).unwrap())
            } else {
                None
            }
        }).collect::<Vec<&T>>()
    }

    pub fn get_mut_with_tag(&mut self, tag: &str) -> Vec<&mut T> {
        self.tags.iter().zip(self.items.iter_mut()).filter_map(|(tags, item)| {
            if tags.contains(&tag) {
                Some(item)
            } else {
                None
            }
        })
        .collect::<Vec<&mut T>>()
    }
}
