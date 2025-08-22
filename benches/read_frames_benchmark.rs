use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::BenchmarkId;
use corrosiff;


use ndarray::prelude::*;

use std::collections::HashMap;

const SHORT_SIFF_PATH: &str = "/Users/stephen/Desktop/Data/imaging/2024-04/2024-04-07/Dh31_LexA_LKir_LGFlamp1/Fly1/BarOnAtTen_1.siff";
const LONG_SIFF_PATH: &str = "/Users/stephen/Desktop/Data/imaging/2024-04/2024-04-17/21Dhh_GCaFLITS/Fly1/Flashes_1.siff";

/// Open multiple files, read either a few frames quickly with and without registration
/// (to compare overhead latency) and then many frames with and without registration
/// (to compare the actual effect of adding registration)
fn criterion_benchmark_read_frames(c: &mut Criterion) {

    let siffreader = corrosiff::open_siff(SHORT_SIFF_PATH).unwrap();
    let mut read_bench = c.benchmark_group("Frame read benchmarks");
    let frame_vec = Vec::<u64>::from_iter(0..40);
    read_bench.bench_with_input(
        BenchmarkId::new("Read short siff, 40 frames unregistered", 
            40,
        ),
        &frame_vec.as_slice(),
        |bench, frames| {
            bench.iter(|| black_box(siffreader.get_frames_intensity(frames, None).unwrap()))
        },
    );

    let mut reg : HashMap<u64, (i32, i32)> = HashMap::new();
    
    frame_vec.iter().for_each(|&x| {
        reg.insert(x, ((x%100) as i32, ((x + 50) % 100) as i32));
    });

    read_bench.bench_with_input(
        BenchmarkId::new("Read short siff, 40 frames registered", 
            40,
        ),
        &frame_vec.as_slice(),
        |bench, frames| {
            bench.iter(|| black_box(siffreader.get_frames_intensity(frames, Some(&reg)).unwrap()))
        },
    );

    let siffreader = corrosiff::open_siff(LONG_SIFF_PATH).unwrap();
    let frame_vec = Vec::<u64>::from_iter(0..49999);
    read_bench.sample_size(10);
    read_bench.bench_with_input(
        BenchmarkId::new("Read long siff, 50k-1 frames unregistered", 
            49999,
        ),
        &frame_vec.as_slice(),
        |bench, frames| {
            bench.iter(|| black_box(siffreader.get_frames_intensity(frames, None).unwrap()))
        },
    );

    let mut reg : HashMap<u64, (i32, i32)> = HashMap::new();
    
    frame_vec.iter().for_each(|&x| {
        reg.insert(x, ((x % 100) as i32, ((x + 50) % 100) as i32 ));
    });

    read_bench.bench_with_input(
        BenchmarkId::new("Read long siff, 50k-1 frames registered", 
            49999,
        ),
        &frame_vec.as_slice(),
        |bench, frames| {
            bench.iter(|| black_box(siffreader.get_frames_intensity(frames, Some(&reg)).unwrap()))
        },
    );

}

fn criterion_benchmark_histograms(c: &mut Criterion) {
    let siffreader = corrosiff::open_siff(SHORT_SIFF_PATH).unwrap();
    let mut read_bench = c.benchmark_group("Histogram benchmarks");
    let frame_vec = Vec::<u64>::from_iter(0..40);
    read_bench.bench_with_input(
        BenchmarkId::new("Read histogram from 40 frames", 
            40,
        ),
        &frame_vec.as_slice(),
        |bench, frames| {
            bench.iter(|| black_box(siffreader.get_histogram(frames).unwrap()))
        },
    );

    let siffreader = corrosiff::open_siff(LONG_SIFF_PATH).unwrap();
    let frame_vec = Vec::<u64>::from_iter(0..siffreader.num_frames() as u64);
    read_bench.sample_size(20);
    read_bench.bench_with_input(
        BenchmarkId::new("Read long siff, get histogram from all frames", 
            -1,
        ),
        &frame_vec.as_slice(),
        |bench, frames| {
            bench.iter(|| black_box(siffreader.get_histogram(frames).unwrap()))
        },
    );

    let dims = siffreader.image_dims().unwrap().to_tuple();
    let mut mask = Array2::<bool>::from_elem(
        (dims.0, dims.1), false
    );
    mask.mapv_inplace(|x| rand::random::<bool>());

    read_bench.bench_with_input(
        BenchmarkId::new("Read long siff, get histogram from all frames with mask", 
            -1,
        ),
        &frame_vec.as_slice(),
        |bench, frames| {
            bench.iter(|| black_box(siffreader.get_histogram_mask(frames, &mask.view(), None).unwrap()))
        },
    );

    let mut reg = corrosiff::RegistrationDict::new();
    frame_vec.iter().for_each(|&x| {
        reg.insert(x, ((x % 123) as i32, ((x + 50) % 87) as i32 ));
    });

    read_bench.bench_with_input(
        BenchmarkId::new("Read long siff, get histogram from all frames with mask and registration", 
            -1,
        ),
        &frame_vec.as_slice(),
        |bench, frames| {
            bench.iter(|| black_box(siffreader.get_histogram_mask(frames, &mask.view(), Some(&reg)).unwrap()))
        },
    );
}

// /// Open multiple files, read either a few frames quickly with and without registration
// /// (to compare overhead latency) and then many frames with and without registration
// /// (to compare the actual effect of adding registration)
// fn criterion_benchmark_tau_d(c: &mut Criterion) {
//     let siffreader = corrosiff::open_siff(SHORT_SIFF_PATH).unwrap();
//     let mut read_bench = c.benchmark_group("Tau_d read benchmarks");

//     let frame_vec = Vec::<u64>::from_iter(0..40);
//     read_bench.bench_with_input(
//         BenchmarkId::new("Read short siff, 40 frames unregistered", 
//             40,
//         ),
//         &frame_vec.as_slice(),
//         |bench, frames| {
//             bench.iter(|| black_box(siffreader.get_frames_tau_d(frames, None).unwrap()))
//         },
//     );

//     let mut reg : HashMap<u64, (i32, i32)> = HashMap::new();
    
//     frame_vec.iter().for_each(|&x| {
//         reg.insert(x, ((x%100) as i32, ((x + 50) % 100) as i32));
//     });

//     read_bench.bench_with_input(
//         BenchmarkId::new("Read short siff, 40 frames registered", 
//             40,
//         ),
//         &frame_vec.as_slice(),
//         |bench, frames| {
//             bench.iter(|| black_box(siffreader.get_frames_tau_d(frames, Some(&reg)).unwrap()))
//         },
//     );

//     let siffreader = corrosiff::open_siff(LONG_SIFF_PATH).unwrap();
//     let frame_vec = Vec::<u64>::from_iter(0..49999);
//     read_bench.sample_size(10);
//     read_bench.bench_with_input(
//         BenchmarkId::new("Read long siff, 50k-1 frames unregistered", 
//             49999,
//         ),
//         &frame_vec.as_slice(),
//         |bench, frames| {
//             bench.iter(|| black_box(siffreader.get_frames_tau_d(frames, None).unwrap()))
//         },
//     );

//     let mut reg : HashMap<u64, (i32, i32)> = HashMap::new();
    
//     frame_vec.iter().for_each(|&x| {
//         reg.insert(x, ((x % 100) as i32, ((x + 50) % 100) as i32 ));
//     });

//     read_bench.bench_with_input(
//         BenchmarkId::new("Read long siff, 50k-1 frames registered", 
//             49999,
//         ),
//         &frame_vec.as_slice(),
//         |bench, frames| {
//             bench.iter(|| black_box(siffreader.get_frames_tau_d(frames, Some(&reg)).unwrap()))
//         },
//     );
// }

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark_read_frames,
    criterion_benchmark_histograms,
    //criterion_benchmark_tau_d,
);
criterion_main!(benches);