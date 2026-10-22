//! A small insertion-ordered map with Python `dict` semantics: assigning an
//! existing key keeps its position, a new key goes last.

#[derive(Clone, Debug, PartialEq)]
pub struct OrderedMap<K, V>(Vec<(K, V)>);

impl<K: PartialEq + Clone, V: Clone> Default for OrderedMap<K, V> {
    fn default() -> Self { Self(Vec::new()) }
}

impl<K: PartialEq + Clone, V: Clone> OrderedMap<K, V> {
    pub fn new() -> Self { Self(Vec::new()) }
    pub fn from_pairs(pairs: impl IntoIterator<Item = (K, V)>) -> Self {
        let mut map = Self::new();
        for (key, value) in pairs { map.insert(key, value); }
        map
    }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn get(&self, key: &K) -> Option<&V> { self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v) }
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> { self.0.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v) }
    pub fn contains_key(&self, key: &K) -> bool { self.0.iter().any(|(k, _)| k == key) }
    pub fn insert(&mut self, key: K, value: V) {
        match self.0.iter_mut().find(|(k, _)| *k == key) { Some(slot) => slot.1 = value, None => self.0.push((key, value)) }
    }
    pub fn set_default(&mut self, key: K, value: V) { if !self.contains_key(&key) { self.0.push((key, value)); } }
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let position = self.0.iter().position(|(k, _)| k == key)?;
        Some(self.0.remove(position).1)
    }
    pub fn keys(&self) -> impl Iterator<Item = &K> { self.0.iter().map(|(k, _)| k) }
    pub fn values(&self) -> impl Iterator<Item = &V> { self.0.iter().map(|(_, v)| v) }
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> { self.0.iter().map(|(k, v)| (k, v)) }
    pub fn update(&mut self, other: &Self) { for (key, value) in other.iter() { self.insert(key.clone(), value.clone()); } }
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &V) -> bool) { self.0.retain(|(k, v)| keep(k, v)); }
    pub fn into_pairs(self) -> Vec<(K, V)> { self.0 }
    pub fn pairs(&self) -> &[(K, V)] { &self.0 }
}
