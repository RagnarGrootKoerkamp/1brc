#![feature(slice_split_once, portable_simd, slice_as_chunks, split_array)]
#![allow(unused)]
use colored::Colorize;
use fxhash::FxHashMap;
use memmap2::Mmap;
use ptr_hash::{PtrHash, PtrHashParams};
use std::{
    env::args,
    io::Read,
    simd::{Simd, SimdPartialEq, ToBitMask},
    vec::Vec,
};

#[derive(Clone)]
#[repr(align(64))]
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
    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum += other.sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
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
    let a = unsafe { *s.get_unchecked(s.len().wrapping_sub(5)) };
    let b = unsafe { *s.get_unchecked(s.len().wrapping_sub(4)) };
    let c = unsafe { *s.get_unchecked(s.len().wrapping_sub(3)) };
    let d = unsafe { *s.get_unchecked(s.len().wrapping_sub(1)) };
    let v = a as V * 1000 * (s.len() >= 5) as V
        + b as V * 100 * (s.len() >= 4) as V
        + c as V * 10
        + d as V;
    if neg {
        -v
    } else {
        v
    }
}

fn format(v: V) -> String {
    format!("{:.1}", v as f64 / 10.0)
}

#[allow(unused)]
fn to_key(name: &[u8]) -> u64 {
    // Hash the first and last 8 bytes.
    let head: [u8; 8] = unsafe { *name.get_unchecked(..8).split_array_ref().0 };
    let tail: [u8; 8] = unsafe {
        *name
            .get_unchecked(name.len().wrapping_sub(8)..)
            .split_array_ref()
            .0
    };
    let shift = 64usize.saturating_sub(8 * name.len());
    let khead = u64::from_ne_bytes(head) << shift;
    let ktail = u64::from_ne_bytes(tail) >> shift;
    khead.wrapping_add(ktail)
}

/// Number of SIMD lanes. AVX2 has 256 bits, so 32 lanes.
const L: usize = 32;
/// The Simd type.
type S = Simd<u8, L>;

/// Find the regions between \n and ; (names) and between ; and \n (values),
/// and calls `callback` for each line.
#[inline(always)]
fn iter_lines<'a>(data: &'a [u8], mut callback: impl FnMut(&'a [u8], &'a [u8])) {
    // TODO: Handle the head and tail.
    let (head, simd_data, _tail) = unsafe { data.align_to::<S>() };
    let data = &data[head.len()..];

    let sep = S::splat(b';');
    let end = S::splat(b'\n');
    let mut start_pos = 0;

    let eq_step = |i: &mut usize| {
        *i += 2;
        let simd = simd_data[*i];
        let eq_sep_l = sep.simd_eq(simd).to_bitmask() as u64;
        let eq_end_l = end.simd_eq(simd).to_bitmask() as u64;
        let simd = simd_data[*i + 1];
        let eq_sep_h = sep.simd_eq(simd).to_bitmask() as u64;
        let eq_end_h = end.simd_eq(simd).to_bitmask() as u64;
        ((eq_sep_h << 32) + eq_sep_l, (eq_end_h << 32) + eq_end_l)
    };
    let i = &mut (-2isize as usize);
    let (mut eq_sep, mut eq_end) = eq_step(i);

    if eq_end.trailing_zeros() < eq_sep.trailing_zeros() {
        let offset = eq_end.trailing_zeros();
        eq_end ^= 1 << offset;
    }

    // TODO: Handle the tail.
    while *i < simd_data.len() - 3 {
        // find ; separator
        if eq_sep == 0 {
            (eq_sep, eq_end) = eq_step(i);
        }
        let offset = eq_sep.trailing_zeros();
        eq_sep ^= 1 << offset;
        let sep_pos = L * *i + offset as usize;

        // find \n newline
        if eq_end == 0 {
            (eq_sep, eq_end) = eq_step(i);
        }
        let offset = eq_end.trailing_zeros();
        eq_end ^= 1 << offset;
        let end_pos = L * *i + offset as usize;

        unsafe {
            let name = data.get_unchecked(start_pos..sep_pos);
            let value = data.get_unchecked(sep_pos + 1..end_pos);
            callback(name, value);
        }

        start_pos = end_pos + 1;
    }
}

fn run(data: &[u8], phf: &PtrHash, num_slots: usize) -> Vec<Record> {
    // Each thread has its own accumulator.
    let mut slots = vec![Record::default(); num_slots];
    iter_lines(data, |name, value| {
        let key = to_key(name);
        let index = phf.index(&key);
        let entry = unsafe { slots.get_unchecked_mut(index) };
        entry.add(parse(value));
    });
    slots
}

fn run_parallel(data: &[u8], phf: &PtrHash, num_slots: usize) -> Vec<Record> {
    let mut slots = std::sync::Mutex::new(vec![Record::default(); num_slots]);

    // Spawn one thread per core.
    let num_threads = std::thread::available_parallelism().unwrap();
    std::thread::scope(|s| {
        let chunks = data.chunks(data.len() / num_threads + 1);
        for chunk in chunks {
            s.spawn(|| {
                // Each thread has its own accumulator.
                let thread_slots = run(chunk, phf, num_slots);

                // Merge results.
                let mut slots = slots.lock().unwrap();
                for (thread_slot, slot) in thread_slots.into_iter().zip(slots.iter_mut()) {
                    slot.merge(&thread_slot);
                }
            });
        }
    });

    slots.into_inner().unwrap()
}

fn to_str(name: &[u8]) -> &str {
    std::str::from_utf8(name).unwrap()
}

#[inline(never)]
fn build_perfect_hash(data: &[u8]) -> (Vec<(u64, &[u8])>, PtrHash, usize) {
    let mut cities_map = FxHashMap::default();

    iter_lines(data, |name, _value| {
        let key = to_key(name);
        let name_in_map = *cities_map.entry(key).or_insert(name);
        assert_eq!(
            name,
            name_in_map,
            "existing = {}  != {} = inserting",
            to_str(name),
            to_str(name_in_map)
        );
    });

    let mut cities = cities_map.into_iter().collect::<Vec<_>>();
    cities.sort_unstable_by_key(|&(_key, name)| name);
    let keys = cities.iter().map(|(k, _)| *k).collect::<Vec<_>>();

    eprintln!("cities {}", keys.len());
    let min_len = cities.iter().map(|x| x.1.len()).min().unwrap();
    let max_len = cities.iter().map(|x| x.1.len()).max().unwrap();
    eprintln!("Min city len: {min_len}");
    eprintln!("Max city len: {max_len}");
    assert!(keys.len() <= 500);

    let num_slots = 2 * cities.len();
    let params = ptr_hash::PtrHashParams {
        alpha: 0.9,
        c: 1.2,
        slots_per_part: num_slots,
        ..PtrHashParams::default()
    };
    eprint!("build \r");
    let start = std::time::Instant::now();
    let ptrhash = PtrHash::new(&keys, params);
    eprintln!(
        "build {}",
        format!("{:>5.1?}", start.elapsed()).bold().green()
    );
    (cities, ptrhash, num_slots)
}

fn main() {
    let start = std::time::Instant::now();
    let filename = &args().nth(1).unwrap_or("measurements.txt".to_string());
    let mut mmap: Mmap;
    let mut data;
    {
        eprint!("mmap  ");
        let mut file = std::fs::File::open(filename).unwrap();
        let start = std::time::Instant::now();
        mmap = unsafe { Mmap::map(&file).unwrap() };
        data = &*mmap;
        eprintln!("{}", format!("{:>5.1?}", start.elapsed()).bold().green());
    }

    // Guaranteed to be aligned for SIMD.
    let offset = unsafe { data.align_to::<S>().0.len() };
    let data = &data[offset..];

    // Build a perfect hash function on the cities found in the first 100k characters.
    let (cities, phf, num_slots) = build_perfect_hash(&data[..100000]);

    let iter_start = std::time::Instant::now();
    let records = run_parallel(data, &phf, num_slots);
    eprintln!(
        "iter:  {}",
        format!("{:>5.2?}", iter_start.elapsed()).bold().green()
    );

    if false {
        for (key, name) in &cities {
            let r = &records[phf.index(key)];
            println!(
                "{}: {}/{}/{}",
                to_str(name),
                format(r.min),
                format(r.avg()),
                format(r.max)
            );
        }
    }

    eprintln!(
        "total: {}",
        format!("{:>5.2?}", start.elapsed()).bold().green()
    );
}
