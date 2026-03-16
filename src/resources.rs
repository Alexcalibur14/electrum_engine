
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle {
    id: usize,
}

impl Handle {
    pub fn new(id: usize) -> Self {
        Handle {
            id,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }
}

#[derive(Clone)]
pub struct Collection<'a, T> {
    current_id: usize,
    items: Vec<T>,
    ids: Vec<usize>,
    tags: Vec<Vec<&'a str>>,
}

impl<'a, T> Collection<'a, T> {
    pub fn new() -> Self {
        Collection {
            current_id: 0,
            items: vec![],
            ids: vec![],
            tags: vec![],
        }
    }

    pub fn push(&mut self, item: T, tags: &[&'a str]) -> Handle {
        let id = self.current_id;
        self.items.push(item);
        self.ids.push(id);
        self.tags.push(tags.to_vec());

        self.current_id += 1;

        Handle::new(id)
    }

    pub fn get(&self, handle: &Handle) -> Option<&T> {
        self.items.get(self.get_index_from_handle(handle).unwrap())
    }

    pub fn get_mut(&mut self, handle: &Handle) -> Option<&mut T> {
        let index = self.get_index_from_handle(handle).unwrap();
        self.items.get_mut(index)
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

    pub fn remove(&mut self, handle: &Handle) {
        let index = self.get_index_from_handle(handle).unwrap();

        self.items.swap_remove(index);
        self.ids.swap_remove(index);
        self.tags.swap_remove(index);
    }

    pub fn clear(&mut self) {
        self.ids.clear();
        self.items.clear();
        self.tags.clear();
    }

    pub fn items(&self) -> &[T] {
        &self.items
    }

    pub fn items_mut(&mut self) -> &mut [T] {
        &mut self.items
    }

    fn get_index_from_handle(&self, handle: &Handle) -> Option<usize> {
        self.ids.iter().position(|id| handle.id == *id)
    }
}

#[derive(Clone, Default)]
pub struct NamedVec<'a, T> {
    items: Vec<T>,
    names: Vec<&'a str>,
}

impl<'a, T> NamedVec<'a, T> {
    pub fn new() -> Self {
        NamedVec {
            items: vec![],
            names: vec![],
        }
    }

    pub fn push(&mut self, item: T, name: &'a str) {
        self.items.push(item);
        self.names.push(name);
    }

    pub fn remove(&mut self, name: &str) -> T {
        let index = self.names.iter().position(|n| *n == name).unwrap();
        self.names.swap_remove(index);
        self.items.swap_remove(index)
    }

    pub fn get(&self, name: &str) -> &T {
        let index = self.names.iter().position(|n| *n == name).unwrap();
        self.items.get(index).unwrap()
    }

    pub fn get_mut(&mut self, name: &str) -> &mut T {
        let index = self.names.iter().position(|n| *n == name).unwrap();
        self.items.get_mut(index).unwrap()
    }

    pub fn items(&self) -> &[T] {
        &self.items
    }

    pub fn items_mut(&mut self) -> &mut [T] {
        &mut self.items
    }

    pub fn clear(&mut self) {
        self.names.clear();
        self.items.clear();
    }
}
