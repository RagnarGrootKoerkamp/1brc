#![feature(slice_split_once, portable_simd, slice_as_chunks)]
use fxhash::{FxHashMap, FxHasher};
use std::{
    env::args,
    hash::Hasher,
    io::Read,
    simd::{Simd, SimdPartialEq, ToBitMask},
};

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
#[allow(unused)]
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

/// Number of SIMD lanes. AVX2 has 256 bits, so 32 lanes.
const L: usize = 32;
/// The Simd type.
type S = Simd<u8, L>;

/// Find the regions between \n and ; (names) and between ; and \n (values),
/// and calls `callback` for each line.
#[inline(always)]
fn iter_lines<'a>(data: &'a [u8], mut callback: impl FnMut(&'a [u8], &'a [u8])) {
    unsafe {
        // TODO: Handle the tail.
        let simd_data: &[S] = data.align_to::<S>().1;

        let sep = S::splat(b';');
        let end = S::splat(b'\n');
        let mut start_pos = 0;
        let mut i = 0;
        let mut eq_sep = sep.simd_eq(simd_data[i]).to_bitmask();
        let mut eq_end = end.simd_eq(simd_data[i]).to_bitmask();

        // TODO: Handle the tail.
        while i < simd_data.len() - 2 {
            // find ; separator
            // TODO if?
            while eq_sep == 0 {
                i += 1;
                eq_sep = sep.simd_eq(simd_data[i]).to_bitmask();
                eq_end = end.simd_eq(simd_data[i]).to_bitmask();
            }
            let offset = eq_sep.trailing_zeros();
            eq_sep ^= 1 << offset;
            let sep_pos = L * i + offset as usize;

            // find \n newline
            // TODO if?
            while eq_end == 0 {
                i += 1;
                eq_sep = sep.simd_eq(simd_data[i]).to_bitmask();
                eq_end = end.simd_eq(simd_data[i]).to_bitmask();
            }
            let offset = eq_end.trailing_zeros();
            eq_end ^= 1 << offset;
            let end_pos = L * i + offset as usize;

            callback(
                data.get_unchecked(start_pos..sep_pos),
                data.get_unchecked(sep_pos + 1..end_pos),
            );

            start_pos = end_pos + 1;
        }
    }
}

fn main() {
    let filename = &args().nth(1).unwrap_or("measurements.txt".to_string());
    let mut data = vec![];
    let offset;
    {
        let stat = std::fs::metadata(filename).unwrap();
        data.reserve(stat.len() as usize + 2 * L);
        // Some hacky stuff to make sure data is aligned to simd lanes.
        data.resize(4 * L, 0);
        let pre_aligned = unsafe { data.align_to::<S>().0 };
        offset = pre_aligned.len();
        assert!(offset < L);
        data.resize(offset, 0);
        let mut file = std::fs::File::open(filename).unwrap();
        file.read_to_end(&mut data).unwrap();
    }
    // Guaranteed to be aligned for SIMD.
    let data = &data[offset..];

    let mut h = FxHashMap::default();

    let callback = |name, value| {
        h.entry(to_key_fx(name))
            .or_insert((Record::default(), name))
            .0
            .add(parse(value));
    };
    iter_lines(data, callback);

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
