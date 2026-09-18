use camerad::ring::{CameraRingWriter, FMT_RGB8, HEADER_BYTES, MAGIC};
use camerad::yuyv;
use std::fs;
use std::path::PathBuf;

fn temp_ring(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("{name}-{}.shm", std::process::id()));
    let _ = fs::remove_file(&p);
    p
}

fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn le64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

struct Header {
    magic: u32,
    fmt: u32,
    width: u32,
    height: u32,
    stride: u32,
    slot_size: u32,
    slots: u32,
    pid: u32,
    seq: u64,
}

fn read_header(path: &PathBuf) -> Header {
    let b = fs::read(path).unwrap();
    Header {
        magic: le32(&b, 0),
        fmt: le32(&b, 4),
        width: le32(&b, 8),
        height: le32(&b, 12),
        stride: le32(&b, 16),
        slot_size: le32(&b, 20),
        slots: le32(&b, 24),
        pid: le32(&b, 28),
        seq: le64(&b, 32),
    }
}

#[test]
fn writer_sets_byte_exact_header() {
    let p = temp_ring("header");
    {
        let w = CameraRingWriter::new(&p, 4, 2, 3).unwrap();
        assert_eq!(w.width(), 4);
        assert_eq!(w.height(), 2);
        assert_eq!(w.stride(), 12);
        assert_eq!(w.slot_size(), 24);
        assert_eq!(w.num_slots(), 3);
    }
    let h = read_header(&p);
    assert_eq!(h.magic, MAGIC);
    assert_eq!(h.fmt, FMT_RGB8);
    assert_eq!(h.width, 4);
    assert_eq!(h.height, 2);
    assert_eq!(h.stride, 12);
    assert_eq!(h.slot_size, 24);
    assert_eq!(h.slots, 3);
    assert_eq!(h.pid, std::process::id());
    assert_eq!(h.seq, 0);
    let _ = fs::remove_file(&p);
}

#[test]
fn frames_rotate_across_slots() {
    let p = temp_ring("rotation");
    {
        let mut w = CameraRingWriter::new(&p, 4, 2, 3).unwrap();
        for (i, val) in [1u8, 2, 3, 4].iter().enumerate() {
            w.submit_with(|s| s.fill(*val));
            assert_eq!(w.seq(), i as u64 + 1);
        }
    }
    let b = fs::read(&p).unwrap();
    let slot_off = |seq: u64| HEADER_BYTES + (seq % 3) as usize * 24;
    assert!(b[slot_off(1)..slot_off(1) + 24].iter().all(|&x| x == 4));
    assert!(b[slot_off(2)..slot_off(2) + 24].iter().all(|&x| x == 2));
    assert!(b[slot_off(3)..slot_off(3) + 24].iter().all(|&x| x == 3));
    assert_eq!(le64(&b, 32), 4);
    let _ = fs::remove_file(&p);
}

#[test]
fn yuyv_frame_lands_in_ring_slot() {
    let p = temp_ring("yuyv");
    let width = 4usize;
    let height = 1usize;
    let yuyv_frame = [82u8, 200, 150, 90, 30, 160, 210, 240];
    {
        let mut w = CameraRingWriter::new(&p, width as u32, height as u32, 3).unwrap();
        w.submit_with(|dst| {
            yuyv::yuyv_to_rgb8(&yuyv_frame, dst, width, height);
        });
    }
    let b = fs::read(&p).unwrap();
    let mut expect = vec![0u8; width * height * 3];
    yuyv::yuyv_to_rgb8(&yuyv_frame, &mut expect, width, height);
    let off = HEADER_BYTES + (width * height * 3);
    assert_eq!(&b[off..off + expect.len()], &expect[..]);
    let _ = fs::remove_file(&p);
}

#[test]
fn rgb24_frame_lands_in_ring_slot() {
    let p = temp_ring("rgb24");
    let width = 4usize;
    let height = 2usize;
    let rgb24_frame: Vec<u8> = (0..(width * height * 3))
        .map(|i| (i * 7 + 3) as u8)
        .collect();
    {
        let mut w = CameraRingWriter::new(&p, width as u32, height as u32, 3).unwrap();
        w.submit_with(|dst| {
            camerad::yuyv::rgb24_to_rgb8(&rgb24_frame, dst, width, height);
        });
    }
    let b = fs::read(&p).unwrap();
    let off = HEADER_BYTES + (width * height * 3);
    assert_eq!(&b[off..off + rgb24_frame.len()], &rgb24_frame[..]);
    let _ = fs::remove_file(&p);
}

#[test]
fn yu12_frame_lands_in_ring_slot() {
    let p = temp_ring("yu12");
    let width = 4usize;
    let height = 2usize;
    let y_size = width * height;
    let uv_size = (width / 2) * (height / 2);
    let mut yu12_frame = vec![0u8; y_size + uv_size * 2];
    for (i, b) in yu12_frame.iter_mut().enumerate() {
        *b = (i * 13 + 5) as u8;
    }
    {
        let mut w = CameraRingWriter::new(&p, width as u32, height as u32, 3).unwrap();
        w.submit_with(|dst| {
            camerad::yuyv::yuv420_to_rgb8(&yu12_frame, dst, width, height);
        });
    }
    let b = fs::read(&p).unwrap();
    let mut expect = vec![0u8; width * height * 3];
    camerad::yuyv::yuv420_to_rgb8(&yu12_frame, &mut expect, width, height);
    let slot_size = width * height * 3;
    let slot = 1u64;
    let off = HEADER_BYTES + (slot % 3) as usize * slot_size;
    assert_eq!(&b[off..off + expect.len()], &expect[..]);
    let _ = fs::remove_file(&p);
}

#[test]
fn rejects_zero_geometry() {
    assert!(CameraRingWriter::new(&temp_ring("zero"), 0, 1, 3).is_err());
    assert!(CameraRingWriter::new(&temp_ring("zero"), 1, 0, 3).is_err());
}
