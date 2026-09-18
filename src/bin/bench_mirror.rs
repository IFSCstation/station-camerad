use std::time::Instant;

fn main() {
    let width = 1280usize;
    let height = 720usize;
    let row_bytes = width * 3;
    let iterations = 500u64;

    let mut row: Vec<u8> = (0..row_bytes).map(|i| (i * 7 + 13) as u8).collect();

    let start = Instant::now();
    for _ in 0..iterations * height as u64 {
        unsafe {
            camerad::rgb24_mirror_row(row.as_mut_ptr(), width);
        }
    }
    let elapsed = start.elapsed();

    let total_bytes = iterations * height as u64 * row_bytes as u64;
    let gbps = total_bytes as f64 / elapsed.as_secs_f64() / 1e9;
    let ns_per_row = elapsed.as_nanos() as f64 / (iterations * height as u64) as f64;

    println!("=== RGB24 mirror NEON benchmark ===");
    println!("Resolution: {width}x{height}");
    println!("Iterations: {iterations}");
    println!("Time: {:.2?}", elapsed);
    println!("Per row: {:.0} ns", ns_per_row);
    println!("Bandwidth: {:.2} GB/s", gbps);
    println!(
        "Estimated CPU at 30fps: {:.1}%",
        ns_per_row * 30.0 * height as f64 / 1e9 * 100.0
    );
}
