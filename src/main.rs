#![feature(slice_split_once, portable_simd, slice_as_chunks, split_array)]
use fxhash::FxHashMap;
use std::{
    env::args,
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
        _ => panic!("Unknown pattern {:?}", to_str(s)),
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

#[allow(unused)]
fn to_key(name: &[u8]) -> u64 {
    // Hash the first and last 8 bytes.
    let head: [u8; 8] = unsafe { *name.get_unchecked(..8).split_array_ref().0 };
    let tail: [u8; 8] = unsafe { *name.get_unchecked(name.len() - 8..).split_array_ref().0 };
    let shift = 64usize.saturating_sub(8 * name.len());
    let khead = u64::from_ne_bytes(head) << shift;
    let ktail = u64::from_ne_bytes(tail) >> shift;
    khead + ktail
}

/// Number of SIMD lanes. AVX2 has 256 bits, so 32 lanes.
const L: usize = 32;
/// The Simd type.
type S = Simd<u8, L>;

/// Find the regions between \n and ; (names) and between ; and \n (values),
/// and calls `callback` for each line.
#[inline(always)]
fn iter_lines<'a>(data: &'a [u8], mut callback: impl FnMut(&'a [u8], &'a [u8])) {
    // TODO: Handle the tail.
    let simd_data: &[S] = unsafe { data.align_to::<S>().1 };

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

        unsafe {
            callback(
                data.get_unchecked(start_pos..sep_pos),
                data.get_unchecked(sep_pos + 1..end_pos),
            );
        }

        start_pos = end_pos + 1;
    }
}

fn to_str(name: &[u8]) -> &str {
    std::str::from_utf8(name).unwrap()
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
        let key = to_key(name);
        let entry = h.entry(key).or_insert((Record::default(), name));
        entry.0.add(parse(value));
        debug_assert_eq!(
            entry.1,
            name,
            "{} != {}\nkey: {key:064b}",
            to_str(entry.1),
            to_str(name)
        );
    };
    iter_lines(data, callback);

    let mut v = h.into_iter().collect::<Vec<_>>();
    v.sort_unstable_by_key(|p| p.1 .1);
    if false {
        for (_key, (r, name)) in &v {
            println!(
                "{}: {}/{}/{}",
                to_str(name),
                format(r.min),
                format(r.avg()),
                format(r.max)
            );
        }
    }
    eprintln!("Num cities: {}", v.len());
    let min_len = v.iter().map(|x| x.1 .1.len()).min().unwrap();
    let max_len = v.iter().map(|x| x.1 .1.len()).max().unwrap();
    eprintln!("Min city len: {min_len}");
    eprintln!("Max city len: {max_len}");
}
