#![feature(
    slice_split_once,
    portable_simd,
    slice_as_chunks,
    split_array,
    type_alias_impl_trait
)]
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
    count: V,
    /// Byte representation of string ~b"bc.d" or ~b"\0c.d".
    min: u32,
    /// Byte representation of string b"bc.d" or b"\0c.d".
    max: u32,
    sum: u64,
}

impl Record {
    fn default() -> Self {
        Self {
            count: 0,
            min: 0,
            max: 0,
            sum: 0,
        }
    }
    fn add(&mut self, raw_value: u32, value: u64) {
        // assert2::debug_assert!(value < 1000);
        self.count += 1;
        self.sum += value;
        self.min = self.min.max(!raw_value);
        self.max = self.max.max(raw_value);
    }
    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum += other.sum_to_val() as u64;
        self.min = self.min.max(other.min);
        self.max = self.max.max(other.max);
    }
    fn sum_to_val(&self) -> V {
        let m = (1 << 21) - 1;
        ((self.sum & m) + 10 * ((self.sum >> 21) & m) + 100 * ((self.sum >> 42) & m)) as _
    }
    /// Return (min, avg, max)
    fn merge_pos_neg(pos: &Record, neg: &Record) -> (V, V, V) {
        let pos_sum = pos.sum as V;
        let neg_sum = neg.sum as V;
        let sum = pos_sum - neg_sum;
        // round to nearest
        let avg = (sum + (pos.count + neg.count) / 2) / (pos.count + neg.count);

        let pos_max = raw_to_value(pos.max);
        let neg_max = -raw_to_value(!neg.min);
        let max = pos_max.max(neg_max);

        let pos_min = raw_to_value(!pos.min);
        let neg_min = -raw_to_value(neg.max);
        let min = pos_min.min(neg_min);

        (min, avg, max)
    }
}

/// Reads raw bytes and masks the ;.
/// Returns something of the form 0x3b3c..3d or 0x003c..3d
fn parse_to_raw(data: &[u8], start: usize, end: usize) -> u32 {
    let raw = u32::from_be_bytes(unsafe { *data.get_unchecked(start..).as_ptr().cast() });
    raw >> (8 * (4 - (end - start)))
}

fn raw_to_pdep(raw: u32) -> u64 {
    //         0b                  bbbb             xxxxcccc     yyyyyyyyyyyydddd // Deposit here
    //         0b                  1111                 1111                 1111 // Mask out trash using &
    let pdep = 0b0000000000000000001111000000000000011111111000001111111111111111u64;
    let mask = 0b0000000000000000001111000000000000000001111000000000000000001111u64;

    let v = unsafe { core::arch::x86_64::_pdep_u64(raw as u64, pdep) };
    v & mask
}

fn raw_to_value(v: u32) -> V {
    let bytes = v.to_be_bytes();
    // s = bc.d
    let b = bytes[0] as V - b'0' as V;
    let c = bytes[1] as V - b'0' as V;
    let d = bytes[3] as V - b'0' as V;
    b as V * 100 * (bytes[0] != 0) as V + c as V * 10 + d as V
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
fn iter_lines<'a>(mut data: &'a [u8], mut callback: impl FnMut(&'a [u8], usize, usize, usize)) {
    // Make sure that the out-of-bounds reads we do are OK.
    data = &data[..data.len() - 32];

    let sep = S::splat(b';');
    let end = S::splat(b'\n');

    let find = |last: usize, sep: S| {
        let simd = S::from_array(unsafe { *data.get_unchecked(last..).as_ptr().cast() });
        let eq = sep.simd_eq(simd).to_bitmask() as u32;
        let offset = eq.trailing_zeros() as usize;
        last + offset
    };

    struct State {
        sep_pos: usize,
        start_pos: usize,
    }
    let init_state = |idx: usize| {
        let first_end = find(idx, end);
        State {
            sep_pos: first_end + 1,
            start_pos: first_end + 1,
        }
    };

    let mut state = init_state(0);

    let mut step = |state: &mut State| {
        state.sep_pos = find(state.sep_pos, sep) + 1;
        let end_pos = find(state.sep_pos, end) + 1;
        assert2::debug_assert!(state.start_pos < state.sep_pos);
        assert2::debug_assert!(state.sep_pos < end_pos);

        // let name = data.get_unchecked(state.start_pos..state.sep_pos - 1);
        // let value = data.get_unchecked(state.sep_pos..end_pos - 1);
        callback(data, state.start_pos, state.sep_pos - 1, end_pos - 1);

        state.start_pos = end_pos;
    };

    while state.start_pos < data.len() {
        step(&mut state);
    }
}

fn run<'a>(data: &'a [u8], phf: &'a PtrHash, num_slots: usize) -> Vec<Record> {
    // Each thread has its own accumulator.
    let mut slots = vec![Record::default(); num_slots];
    iter_lines(data, |data, start, mut sep, end| {
        unsafe {
            // If value is negative, extend name by one character.
            sep += (data.get_unchecked(sep + 1) == &b'-') as usize;
            let name = data.get_unchecked(start..sep);
            let key = to_key(name);
            let index = phf.index_single_part(&key);
            let entry = slots.get_unchecked_mut(index);
            let raw = parse_to_raw(data, sep + 1, end);
            // We use the raw for min/max purposes.
            entry.add(raw, raw_to_pdep(raw));
        }
    });
    slots
}

fn run_parallel(data: &[u8], phf: &PtrHash, num_slots: usize, num_threads: usize) -> Vec<Record> {
    if num_threads == 0 {
        return run(data, phf, num_slots);
    }

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
fn build_perfect_hash(data: &[u8]) -> (Vec<Vec<u8>>, PtrHash, usize) {
    let mut cities_map = FxHashMap::default();

    iter_lines(data, |data, start, sep, _end| {
        let name = unsafe { data.get_unchecked(start..sep) };
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
        let name = unsafe { data.get_unchecked(start..sep + 1) };
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
    let names = cities
        .iter()
        .map(|(_, name)| name.to_vec())
        .collect::<Vec<_>>();

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
    (names, ptrhash, num_slots)
}

#[derive(clap::Parser)]
struct Args {
    input: Option<String>,

    #[clap(short = 'j', long)]
    threads: Option<usize>,

    #[clap(long)]
    print: bool,
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
    let (names, phf, num_slots) = build_perfect_hash(&data[..100000]);

    let records = run_parallel(
        data,
        &phf,
        num_slots,
        args.threads
            .unwrap_or(available_parallelism().unwrap().into()),
    );

    if args.print {
        for name in &names {
            if *name.last().unwrap() != b';' {
                continue;
            }
            let namepos = &name[..name.len() - 1];
            let kpos = to_key(namepos);
            let kneg = to_key(name);

            let idxpos = phf.index_single_part(&kpos);
            let idxneg = phf.index_single_part(&kneg);
            let rpos = &records.get(idxpos).unwrap();
            let rneg = &records.get(idxneg).unwrap();
            let (min, avg, max) = Record::merge_pos_neg(rpos, rneg);
            eprintln!(
                "{}: {}/{}/{}",
                to_str(namepos),
                format(min),
                format(avg),
                format(max)
            );
        }
    }

    eprintln!(
        "total: {}",
        format!("{:>5.2?}", start.elapsed()).bold().green()
    );
}

#[cfg(test)]
mod test {
    #[test]
    fn parse_raw() {
        use super::*;
        let d = b"12.3";
        let raw = parse_to_raw(d, 0, 4);
        let v = raw_to_value(raw);
        assert_eq!(v, 123);

        let d = b"12.3";
        let raw = parse_to_raw(d, 1, 4);
        let v = raw_to_value(raw);
        assert_eq!(v, 23);
    }
}
