#![feature(
    slice_split_once,
    portable_simd,
    slice_as_chunks,
    type_alias_impl_trait,
    int_roundings
)]
use clap::Parser;
use colored::Colorize;
use fxhash::FxHashSet;
use memmap2::Mmap;
use ptr_hash::PtrHashParams;
use std::{
    simd::{cmp::SimdPartialEq, Simd},
    thread::available_parallelism,
    vec::Vec,
};

type V = i32;

type PtrHash = ptr_hash::DefaultPtrHash<ptr_hash::hash::FxHash, u64>;

struct Phf {
    ptr_hash: PtrHash,
    keys: Vec<Vec<u8>>,
    slots: Vec<Record>,
}

impl Phf {
    fn new(mut keys: Vec<Vec<u8>>) -> Self {
        keys.sort();

        let num_slots = keys.len() * 5 / 2;
        let params = ptr_hash::PtrHashParams {
            alpha: 0.9,
            c: 1.5,
            slots_per_part: num_slots,
            ..PtrHashParams::default()
        };

        let mut hashes: Vec<u64> = keys.iter().map(|key| hash_name(key)).collect();
        hashes.sort();
        for (x, y) in hashes.iter().zip(hashes.iter().skip(1)) {
            assert!(*x != *y, "DUPLICATE HASH");
        }

        let ptr_hash = PtrHash::new(&hashes, params);

        let slots = vec![Record::default(); num_slots];

        Self {
            ptr_hash,
            keys,
            slots,
        }
    }
    fn compute_index(&self, hash: u64) -> usize {
        self.ptr_hash.index_single_part(&hash)
    }
    fn get_index_mut(&mut self, idx: usize) -> &mut Record {
        &mut self.slots[idx]
    }
    fn index_hash_mut(&mut self, hash: u64) -> &mut Record {
        &mut self.slots[self.ptr_hash.index_single_part(&hash)]
    }
    fn index<'b>(&'b self, key: &[u8]) -> &'b Record {
        let hash = hash_name(key);
        &self.slots[self.compute_index(hash)]
    }
    fn index_mut<'b>(&'b mut self, key: &[u8]) -> &'b mut Record {
        self.index_hash_mut(hash_name(key))
    }
    fn merge(&mut self, r: Self) {
        // TODO: If key sets are equal or one is a subset of the other, merge
        // smaller into larger.
        let mut new_keys = vec![];
        let mut i1 = 0;
        let mut i2 = 0;
        while i1 < self.keys.len() && i2 < r.keys.len() {
            if self.keys[i1] == r.keys[i2] {
                new_keys.push(self.keys[i1].clone());
                i1 += 1;
                i2 += 1;
                continue;
            }
            if self.keys[i1] < r.keys[i2] {
                new_keys.push(self.keys[i1].clone());
                i1 += 1;
                continue;
            }
            if self.keys[i1] > r.keys[i2] {
                new_keys.push(r.keys[i2].clone());
                i2 += 1;
                continue;
            }
            panic!();
        }
        while i1 < self.keys.len() {
            new_keys.push(self.keys[i1].clone());
            i1 += 1;
        }
        while i2 < r.keys.len() {
            new_keys.push(r.keys[i2].clone());
            i2 += 1;
        }
        let mut new_phf = Self::new(new_keys);
        for key in &self.keys {
            new_phf.index_mut(key).merge(self.index(key));
        }
        for key in &r.keys {
            new_phf.index_mut(key).merge(r.index(key));
        }
        *self = new_phf;
    }
}

#[derive(Clone, Debug)]
#[repr(align(32))]
struct Record {
    count: u64,
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
        // See https://en.algorithmica.org/hpc/algorithms/argmin/
        if raw_value < self.min {
            self.min = raw_value;
        }
        if raw_value > self.max {
            self.max = raw_value;
        }
    }
    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum += other.sum_to_val() as u64;
        self.min = self.min.min(other.min);
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
        let count = (pos.count + neg.count) as V;
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
    raw >> (8 * (4 - (end - start)))
}

fn raw_to_pdep(raw: u32) -> u64 {
    #[cfg(feature = "no_pdep")]
    {
        let raw = raw as u64;
        (raw & 15) | ((raw & (15 << 16)) << (21 - 16)) | ((raw & (15 << 24)) << (42 - 24))
    }
    #[cfg(not(feature = "no_pdep"))]
    {
        let mask = 0x0f0f000f;
        let raw = raw & mask;
        // input                                     0011bbbb0011cccc........0011dddd
        //         0b                  bbbb             xxxxcccc     yyyyyyyyyyyydddd // Deposit here
        //         0b                  1111                 1111                 1111 // Mask out trash using &
        let pdep = 0b0000000000000000001111000000000000011111111000001111111111111111u64;
        unsafe { core::arch::x86_64::_pdep_u64(raw as u64, pdep) }
    }
}

fn raw_to_value(v: u32) -> V {
    let mask = 0x0f0f000f;
    let bytes = (v & mask).to_be_bytes();
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
fn hash_name(name: &[u8]) -> u64 {
    // Hash the first and last 8 bytes.
    // TODO: More robust hash that actually uses all characters.
    let head: [u8; 8] = unsafe { *name.get_unchecked(..8).split_first_chunk().unwrap().0 };
    let tail: [u8; 8] = unsafe {
        *name
            .get_unchecked(name.len().wrapping_sub(8)..)
            .split_first_chunk()
            .unwrap()
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

#[derive(Copy, Clone)]
struct State {
    start: usize,
    sep: usize,
    end: usize,
}

/// Find the regions between \n and ; (names) and between ; and \n (values),
/// and calls `callback` for each line.
#[inline(always)]
fn iter_lines<'a>(
    mut data: &'a [u8],
    mut callback: impl FnMut(&'a [u8], State, State, State, State),
) {
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

    let init_state = |idx: usize| {
        let first_end = find_long(idx, end);
        State {
            start: first_end + 1,
            sep: first_end + 1,
            end: 0,
        }
    };

    let mut state0 = init_state(0);
    let mut state1 = init_state(data.len() / 4);
    let mut state2 = init_state(2 * data.len() / 4);
    let mut state3 = init_state(3 * data.len() / 4);

    // Duplicate each line for each input state.
    macro_rules! step {
        [$($s:expr),*] => {
            $($s.sep = find_long($s.sep + 1, sep) ;)*
                $($s.end = find($s.sep + 1, end) ;)*
                callback(data, $($s, )*);
                $($s.start = $s.end + 1;)*
        }
    }

    while state3.start < data.len() {
        step!(state0, state1, state2, state3);
    }
}

fn run(data: &[u8], keys: &[Vec<u8>]) -> Phf {
    // Each thread has its own accumulator.
    let mut h = Phf::new(keys.to_vec());
    iter_lines(
        data,
        |data, mut s0: State, mut s1: State, mut s2: State, mut s3: State| {
            unsafe {
                // If value is negative, extend name by one character.
                s0.sep += (data.get_unchecked(s0.sep + 1) == &b'-') as usize;
                let name0 = data.get_unchecked(s0.start..s0.sep);

                s1.sep += (data.get_unchecked(s1.sep + 1) == &b'-') as usize;
                let name1 = data.get_unchecked(s1.start..s1.sep);

                s2.sep += (data.get_unchecked(s2.sep + 1) == &b'-') as usize;
                let name2 = data.get_unchecked(s2.start..s2.sep);

                s3.sep += (data.get_unchecked(s3.sep + 1) == &b'-') as usize;
                let name3 = data.get_unchecked(s3.start..s3.sep);

                let raw0 = parse_to_raw(data, s0.sep + 1, s0.end);
                let raw1 = parse_to_raw(data, s1.sep + 1, s1.end);
                let raw2 = parse_to_raw(data, s2.sep + 1, s2.end);
                let raw3 = parse_to_raw(data, s3.sep + 1, s3.end);

                let h0 = hash_name(name0);
                let h1 = hash_name(name1);
                let h2 = hash_name(name2);
                let h3 = hash_name(name3);

                let idx0 = h.compute_index(h0);
                let idx1 = h.compute_index(h1);
                let idx2 = h.compute_index(h2);
                let idx3 = h.compute_index(h3);

                h.get_index_mut(idx0).add(raw0, raw_to_pdep(raw0));
                h.get_index_mut(idx1).add(raw1, raw_to_pdep(raw1));
                h.get_index_mut(idx2).add(raw2, raw_to_pdep(raw2));
                h.get_index_mut(idx3).add(raw3, raw_to_pdep(raw3));
            }
        },
    );
    h
}

fn run_parallel(data: &[u8], keys: &[Vec<u8>], num_threads: usize) -> Phf {
    if num_threads == 0 {
        return run(data, keys);
    }

    let phf = std::sync::Mutex::new(Phf::new(keys.to_vec()));

    // Spawn one thread per core.
    std::thread::scope(|s| {
        let chunks = data.chunks(data.len() / num_threads + 1);
        for chunk in chunks {
            s.spawn(|| {
                // Each thread has its own accumulator.
                let thread_phf = run(chunk, keys);

                // Merge results.
                phf.lock().unwrap().merge(thread_phf);
            });
        }
    });

    phf.into_inner().unwrap()
}

fn to_str(name: &[u8]) -> &str {
    std::str::from_utf8(name).unwrap()
}

/// Returns a list of city names found in data.
/// Each city is returned twice, once as `<city>` and once as `<city>;`,
/// with the latter being used to accumulate negative temperatures.
#[inline(never)]
fn find_city_names(data: &[u8]) -> Vec<Vec<u8>> {
    let mut cities = FxHashSet::default();

    let mut callback = |data: &[u8], state: State| {
        let State { start, sep, .. } = state;
        let name = unsafe { data.get_unchecked(start..sep) };
        cities.insert(name.to_vec());

        // Do the same for the name with ; appended.
        let name = unsafe { data.get_unchecked(start..sep + 1) };
        cities.insert(name.to_vec());
    };
    iter_lines(data, |d, s0, s1, s2, s3| {
        flatten_callback(d, s0, s1, s2, s3, &mut callback)
    });

    let mut cities: Vec<_> = cities.into_iter().collect();
    cities.sort();
    cities
}

fn flatten_callback<'a>(
    data: &'a [u8],
    s0: State,
    s1: State,
    s2: State,
    s3: State,
    callback: &mut impl FnMut(&'a [u8], State),
) {
    callback(data, s0);
    callback(data, s1);
    callback(data, s2);
    callback(data, s3);
}

#[derive(clap::Parser)]
struct Args {
    input: Option<String>,

    #[clap(short = 'j', long)]
    threads: Option<usize>,

    #[clap(long)]
    print: bool,

    #[clap(long)]
    stats: bool,
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
    let names = find_city_names(&data[..4000000]);

    if args.stats {
        eprintln!("Num cities: {}", names.len());
        let mut lens = vec![0; 102];
        for n in &names {
            if *n.last().unwrap() == b';' {
                continue;
            }
            lens[n.len()] += 1;
        }
        for (len, count) in lens.iter().enumerate() {
            if *count != 0 {
                eprintln!("{}: {}", len, count);
            }
        }
    }

    let phf = run_parallel(
        data,
        &names,
        args.threads
            .unwrap_or(available_parallelism().unwrap().into()),
    );

    if args.print {
        print!("{{");
        let mut first = true;

        let mut keys = phf.keys.clone();
        keys.sort_by(|kl, kr| to_str(kl).cmp(to_str(kr)));

        for name in &keys {
            if *name.last().unwrap() != b';' {
                continue;
            }
            let namepos = &name[..name.len() - 1];

            let rpos = phf.index(namepos);
            let rneg = phf.index(name);
            let (min, avg, max) = Record::merge_pos_neg(rpos, rneg);

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
