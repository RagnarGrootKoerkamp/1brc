#![feature(
    slice_split_once,
    portable_simd,
    slice_as_chunks,
    split_array,
    type_alias_impl_trait,
    int_roundings
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
    // count: V,
    // Storing these as two u32 is nice, because they are read as a single u64.
    /// Byte representation of string ~b"bc.d" or ~b"\0c.d".
    min: u32,
    /// Byte representation of string b"bc.d" or b"\0c.d".
    max: u32,
    sum: u64,
}

impl Record {
    fn default() -> Self {
        Self {
            // count: 0,
            min: 0,
            max: 0,
            sum: 0,
        }
    }
    fn add(&mut self, raw_value: u32, value: u64) {
        // assert2::debug_assert!(value < 1000);
        // self.count += 1;
        self.sum += value;
        // See https://en.algorithmica.org/hpc/algorithms/argmin/
        if raw_value < self.min {
            self.min = raw_value;
        }
        if raw_value > self.max {
            self.max = raw_value;
        }
    }
    fn merge(&mut self, other: &Self) {
        // self.count += other.count;
        self.sum += other.sum_to_val() as u64;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
    fn sum_to_val(&self) -> V {
        let m = (1 << 21) - 1;
        ((self.sum & m) + 10 * ((self.sum >> 21) & m) + 100 * ((self.sum >> 42) & m)) as _
    }
    /// Return (min, avg, max)
    fn merge_pos_neg(pos: &Record, neg: &Record, avg_count: usize) -> (V, V, V) {
        let pos_sum = pos.sum as V;
        let neg_sum = neg.sum as V;
        let sum = pos_sum - neg_sum;
        // let count = pos.count + neg.count;
        let count = avg_count as V;
        // round to nearest
        let avg = (sum + count / 2).div_floor(count);

        let pos_max = raw_to_value(pos.max);
        let neg_max = -raw_to_value(neg.min);
        let max = pos_max.max(neg_max);

        let pos_min = raw_to_value(pos.min);
        let neg_min = -raw_to_value(neg.max);
        let min = pos_min.min(neg_min);

        (min, avg, max)
    }
}

/// Reads raw bytes and masks the ; and the b'0'=0x30.
/// Returns something of the form 0x0b0c..0d or 0x000c..0d
fn parse_to_raw(data: &[u8], start: usize, end: usize) -> u32 {
    let raw = u32::from_be_bytes(unsafe { *data.get_unchecked(start..).as_ptr().cast() });
    let raw = raw >> (8 * (4 - (end - start)));
    let mask = 0x0f0f000f;
    raw & mask
}

fn raw_to_pdep(raw: u32) -> u64 {
    // input                                     0011bbbb0011cccc........0011dddd
    //         0b                  bbbb             xxxxcccc     yyyyyyyyyyyydddd // Deposit here
    //         0b                  1111                 1111                 1111 // Mask out trash using &
    let pdep = 0b0000000000000000001111000000000000011111111000001111111111111111u64;
    unsafe { core::arch::x86_64::_pdep_u64(raw as u64, pdep) }
}

fn raw_to_value(v: u32) -> V {
    let bytes = v.to_be_bytes();
    // s = bc.d
    let b = bytes[0] as V;
    let c = bytes[1] as V;
    let d = bytes[3] as V;
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
    // Modified to be able to search regions longer than 32.
    let find_long = |mut last: usize, sep: S| {
        let simd = S::from_array(unsafe { *data.get_unchecked(last..).as_ptr().cast() });
        let mut eq = sep.simd_eq(simd).to_bitmask() as u32;
        if eq == 0 {
            while eq == 0 {
                last += 32;
                let simd = S::from_array(unsafe { *data.get_unchecked(last..).as_ptr().cast() });
                eq = sep.simd_eq(simd).to_bitmask() as u32;
            }
        }
        let offset = eq.trailing_zeros() as usize;
        last + offset
    };

    struct State {
        start_pos: usize,
        sep_pos: usize,
        end_pos: usize,
    }
    let init_state = |idx: usize| {
        let first_end = find(idx, end);
        State {
            start_pos: first_end + 1,
            sep_pos: first_end + 1,
            end_pos: 0,
        }
    };

    let mut state = init_state(0);

    let mut step = |state: &mut State| {
        state.sep_pos = find_long(state.sep_pos, sep) + 1;
        state.end_pos = find(state.sep_pos, end) + 1;
        callback(data, state.start_pos, state.sep_pos - 1, state.end_pos - 1);
        state.start_pos = state.end_pos;
    };

    while state.start_pos < data.len() {
        step(&mut state);
    }
}

fn run<'a>(data: &'a [u8], phf: &'a PtrHash, num_slots: usize) -> (Vec<Record>, usize) {
    // Each thread has its own accumulator.
    let mut slots = vec![Record::default(); num_slots];
    let mut num_records = 0;
    iter_lines(data, |data, start, mut sep, end| {
        num_records += 1;
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
    (slots, num_records)
}

fn run_parallel(
    data: &[u8],
    phf: &PtrHash,
    num_slots: usize,
    num_threads: usize,
) -> (Vec<Record>, usize) {
    if num_threads == 0 {
        return run(data, phf, num_slots);
    }

    let slots = std::sync::Mutex::new(vec![Record::default(); num_slots]);
    let num_records = std::sync::Mutex::new(0);

    // Spawn one thread per core.
    std::thread::scope(|s| {
        let chunks = data.chunks(data.len() / num_threads + 1);
        for chunk in chunks {
            s.spawn(|| {
                // Each thread has its own accumulator.
                let (thread_slots, thread_num_records) = run(chunk, phf, num_slots);

                // Merge results.
                *num_records.lock().unwrap() += thread_num_records;
                let mut slots = slots.lock().unwrap();
                for (thread_slot, slot) in thread_slots.into_iter().zip(slots.iter_mut()) {
                    slot.merge(&thread_slot);
                }
            });
        }
    });

    (
        slots.into_inner().unwrap(),
        num_records.into_inner().unwrap(),
    )
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
    // assert!(keys.len() <= 1000);

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

    let (records, num_records) = run_parallel(
        data,
        &phf,
        num_slots,
        args.threads
            .unwrap_or(available_parallelism().unwrap().into()),
    );

    if args.print {
        let records_per_city = num_records / names.len();

        print!("{{");
        let mut first = true;
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
            let (min, avg, max) = Record::merge_pos_neg(rpos, rneg, records_per_city);

            if !first {
                print!(", ");
            }
            first = false;

            print!(
                "{}={}/{}/{}",
                to_str(namepos),
                format(min),
                format(avg),
                format(max)
            );
        }
        println!("}}");
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
