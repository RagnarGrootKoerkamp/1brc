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
    let mut data = String::new();
    {
        let mut file = std::fs::File::open(filename).unwrap();
        file.read_to_string(&mut data).unwrap();
    }
    let mut h = HashMap::new();
    for line in data.trim().split('\n') {
        let (name, value) = line.split_once(';').unwrap();
        let value = value.parse::<f32>().unwrap();
        h.entry(name).or_insert(Record::default()).add(value);
    }

    let mut v = h.into_iter().collect::<Vec<_>>();
    v.sort_unstable_by_key(|p| p.0);
    for (name, r) in v {
        println!("{name}: {:.1}/{:.1}/{:.1}", r.min, r.avg(), r.max);
    }
}
