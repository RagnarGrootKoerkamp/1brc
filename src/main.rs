#![feature(slice_split_once)]
use std::{collections::HashMap, env::args, io::Read};

struct Record {
    count: u32,
    min: f32,
    max: f32,
    sum: f32,
}

impl Record {
    fn default() -> Self {
        Self {
            count: 0,
            min: 1000.0,
            max: -1000.0,
            sum: 0.0,
        }
    }
    fn add(&mut self, value: f32) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }
    fn avg(&self) -> f32 {
        self.sum / self.count as f32
    }
}

fn main() {
    let filename = args().nth(1).unwrap_or("measurements.txt".to_string());
    let mut data = vec![];
    {
        let mut file = std::fs::File::open(filename).unwrap();
        file.read_to_end(&mut data).unwrap();
        assert!(data.pop() == Some(b'\n'));
    }
    let mut h = HashMap::new();
    for line in data.split(|&c| c == b'\n') {
        let (name, value) = line.split_once(|&c| c == b';').unwrap();
        let value = unsafe { std::str::from_utf8_unchecked(value) }
            .parse::<f32>()
            .unwrap();
        h.entry(name).or_insert(Record::default()).add(value);
    }

    let mut v = h.into_iter().collect::<Vec<_>>();
    v.sort_unstable_by_key(|p| p.0);
    for (name, r) in &v {
        println!(
            "{}: {:.1}/{:.1}/{:.1}",
            std::str::from_utf8(name).unwrap(),
            r.min,
            r.avg(),
            r.max
        );
    }
    eprintln!("Num records: {}", v.len());
}
