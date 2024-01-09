#![feature(slice_split_once, portable_simd, slice_as_chunks, split_array)]
use clap::Parser;
use colored::Colorize;
use fxhash::FxHashMap;
use memmap2::Mmap;
use ptr_hash::PtrHashParams;
use std::{
    simd::{cmp::SimdPartialEq, Simd},
    thread::available_parallelism,
    vec::Vec,
};

type V = i32;

type PtrHash = ptr_hash::DefaultPtrHash<ptr_hash::hash::FxHash, u64>;

#[derive(Clone)]
#[repr(align(32))]
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
            min: V::MAX,
            max: V::MIN,
            sum: 0,
        }
    }
    fn add(&mut self, value: V) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.max(-value);
        self.max = self.max.max(value);
    }
    fn avg(&self) -> V {
        self.sum / self.count as V
    }
    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum += other.sum;
        self.min = self.min.max(other.min);
        self.max = self.max.max(other.max);
    }
}

fn parse(s: &[u8]) -> V {
    debug_assert!(s[0] != b'-');
    // s = bc.d
    let b = unsafe { *s.get_unchecked(s.len().wrapping_sub(4)) - b'0' };
    let c = unsafe { *s.get_unchecked(s.len().wrapping_sub(3)) - b'0' };
    let d = unsafe { *s.get_unchecked(s.len().wrapping_sub(1)) - b'0' };
    b as V * 100 * (s.len() >= 4) as V + c as V * 10 + d as V
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
fn iter_lines<'a>(mut data: &'a [u8], mut callback: impl FnMut(&'a [u8], &'a [u8])) {
    // Make sure that the out-of-bounds reads we do are OK.
    data = &data[..data.len() - 32];

    let sep = S::splat(b';');
    let end = S::splat(b'\n');

    let find = |last: usize, sep: S| {
        let simd = S::from_array(unsafe { *data.get_unchecked(last..).as_ptr().cast() });
        let eq = sep.simd_eq(simd).to_bitmask();
        let offset = eq.trailing_zeros() as usize;
        last + offset
    };

    struct State {
        sep_pos: usize,
        start_pos: usize,
    }
    let init_state = |idx: usize| {
        let first_end = idx + data[idx..].iter().position(|&c| c == b'\n').unwrap();
        State {
            sep_pos: first_end,
            start_pos: first_end,
        }
    };

    let mut state = init_state(0);

    let mut step = |state: &mut State| {
        state.sep_pos = find(state.sep_pos, sep) + 1;
        let end_pos = find(state.sep_pos, end) + 1;

        unsafe {
            let name = data.get_unchecked(state.start_pos..state.sep_pos - 1);
            let value = data.get_unchecked(state.sep_pos..end_pos - 1);
            callback(name, value);
        }

        state.start_pos = end_pos;
    };

    while state.start_pos < data.len() {
        step(&mut state);
    }
}

fn run(data: &[u8], phf: &PtrHash, num_slots: usize) -> Vec<Record> {
    // Each thread has its own accumulator.
    let mut slots = vec![Record::default(); num_slots];
    iter_lines(data, |mut name, mut value| {
        // If value is negative, extend name by one character.
        unsafe {
            if value.get_unchecked(0) == &b'-' {
                value = value.get_unchecked(1..);
                name = name.get_unchecked(..name.len() + 1);
            }
        }
        let key = to_key(name);
        let index = phf.index_single_part(&key);
        let entry = unsafe { slots.get_unchecked_mut(index) };
        entry.add(parse(value));
    });
    slots
}

fn run_parallel(data: &[u8], phf: &PtrHash, num_slots: usize, num_threads: usize) -> Vec<Record> {
    let slots = std::sync::Mutex::new(vec![Record::default(); num_slots]);

    // Spawn one thread per core.
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
        // Do the same for the name with ; appended.
        let name = unsafe { name.get_unchecked(..name.len() + 1) };
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

    // eprintln!("cities {}", keys.len());
    // let min_len = cities.iter().map(|x| x.1.len()).min().unwrap();
    // let max_len = cities.iter().map(|x| x.1.len()).max().unwrap();
    // eprintln!("Min city len: {min_len}");
    // eprintln!("Max city len: {max_len}");
    assert!(keys.len() <= 1000);

    let num_slots = 2 * cities.len();
    let params = ptr_hash::PtrHashParams {
        alpha: 0.9,
        c: 1.5,
        slots_per_part: num_slots,
        ..PtrHashParams::default()
    };
    let ptrhash = PtrHash::new(&keys, params);
    (cities, ptrhash, num_slots)
}

#[derive(clap::Parser)]
struct Args {
    input: Option<String>,

    #[clap(short = 'j', long)]
    threads: Option<usize>,
}

fn main() {
    let args = Args::parse();

    let start = std::time::Instant::now();
    let filename = args.input.unwrap_or("measurements.txt".to_string());
    let mmap: Mmap;
    let data;
    {
        let file = std::fs::File::open(filename).unwrap();
        mmap = unsafe { Mmap::map(&file).unwrap() };
        data = &*mmap;
    }

    // Guaranteed to be aligned for SIMD.
    let offset = unsafe { data.align_to::<S>().0.len() };
    let data = &data[offset..];

    // Build a perfect hash function on the cities found in the first 100k characters.
    let (cities, phf, num_slots) = build_perfect_hash(&data[..100000]);

    let records = run_parallel(
        data,
        &phf,
        num_slots,
        args.threads
            .unwrap_or(available_parallelism().unwrap().into()),
    );

    if false {
        for (key, name) in &cities {
            let r = &records[phf.index_single_part(key)];
            println!(
                "{}: {}/{}/{}",
                to_str(name),
                format(-r.min),
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
