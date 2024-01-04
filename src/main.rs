#![feature(slice_split_once)]
use fxhash::{FxHashMap, FxHasher};
use memchr::arch::x86_64::avx2::memchr::One;
use std::{env::args, hash::Hasher, io::Read};

struct Record {
    count: u32,
    min: V,
    max: V,
    sum: V,
}

impl Record {
    fn default() -> Self {
        Self {
            count: 0,
            min: i32::MAX,
            max: i32::MIN,
            sum: 0,
        }
    }
    fn add(&mut self, value: V) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }
    fn avg(&self) -> V {
        self.sum / self.count as V
    }
}

type V = i32;

fn parse(mut s: &[u8]) -> V {
    let neg = unsafe {
        if *s.get_unchecked(0) == b'-' {
            s = s.get_unchecked(1..);
            true
        } else {
            false
        }
    };
    // s = abc.d
    let (a, b, c, d) = match s {
        [c, b'.', d] => (0, 0, c - b'0', d - b'0'),
        [b, c, b'.', d] => (0, b - b'0', c - b'0', d - b'0'),
        [a, b, c, b'.', d] => (a - b'0', b - b'0', c - b'0', d - b'0'),
        [c] => (0, 0, 0, c - b'0'),
        [b, c] => (0, b - b'0', c - b'0', 0),
        [a, b, c] => (a - b'0', b - b'0', c - b'0', 0),
        _ => panic!("Unknown pattern {:?}", std::str::from_utf8(s).unwrap()),
    };
    let v = a as V * 1000 + b as V * 100 + c as V * 10 + d as V;
    if neg {
        -v
    } else {
        v
    }
}

fn format(v: V) -> String {
    format!("{:.1}", v as f64 / 10.0)
}

fn to_key_fx(name: &[u8]) -> u64 {
    let mut h = FxHasher::default();
    h.write(name);
    h.finish()
}
fn to_key(name: &[u8]) -> u64 {
    let mut key = [0u8; 8];
    let l = name.len().min(8);
    unsafe {
        key.get_unchecked_mut(..l)
            .copy_from_slice(name.get_unchecked(..l));
    }
    let k = u64::from_ne_bytes(key);
    k ^ name.len() as u64
}

fn main() {
    let filename = &args().nth(1).unwrap_or("measurements.txt".to_string());
    let mut data = vec![];
    {
        let stat = std::fs::metadata(filename).unwrap();
        data.reserve(stat.len() as usize + 1);
        let mut file = std::fs::File::open(filename).unwrap();
        file.read_to_end(&mut data).unwrap();
    }
    let mut h = FxHashMap::default();
    let mut data = &data[..];
    let sep = One::new(b';').unwrap();
    let newline = One::new(b'\n').unwrap();
    unsafe {
        while !data.is_empty() {
            let separator = sep.find(data).unwrap();
            let end = newline.find(data.get_unchecked(separator..)).unwrap();
            let name = data.get_unchecked(..separator);
            let value = data.get_unchecked(separator + 1..separator + end);
            h.entry(to_key_fx(name))
                .or_insert((Record::default(), name))
                .0
                .add(parse(value));
            data = data.get_unchecked(separator + end + 1..);
        }
    }

    let mut v = h.into_iter().collect::<Vec<_>>();
    v.sort_unstable_by_key(|p| p.0);
    for (_key, (r, name)) in &v {
        println!(
            "{}: {}/{}/{}",
            std::str::from_utf8(name).unwrap(),
            format(r.min),
            format(r.avg()),
            format(r.max)
        );
    }
    eprintln!("Num cities: {}", v.len());
    let min_len = v.iter().map(|x| x.1 .1.len()).min().unwrap();
    let max_len = v.iter().map(|x| x.1 .1.len()).max().unwrap();
    eprintln!("Min city len: {min_len}");
    eprintln!("Max city len: {max_len}");
}
