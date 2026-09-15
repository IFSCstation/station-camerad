use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::compiler_fence;
use std::sync::atomic::Ordering;

use libc::{MAP_SHARED, PROT_READ, PROT_WRITE};

pub const CAM_RING_DEFAULT: &str = "/dev/shm/body_estim_camera";
pub const HEADER_BYTES: usize = 64;
pub const FMT_RGB8: u32 = 1;
pub const MAGIC: u32 = 0xCAC0_0001;

const OFF_MAGIC: usize = 0;
const OFF_FMT: usize = 4;
const OFF_WIDTH: usize = 8;
const OFF_HEIGHT: usize = 12;
const OFF_STRIDE: usize = 16;
const OFF_SLOT_SIZE: usize = 20;
const OFF_SLOTS: usize = 24;
const OFF_PID: usize = 28;
const OFF_SEQ: usize = 32;

pub struct CameraRingWriter {
    ptr: *mut u8,
    len: usize,
    _file: File,
    width: u32,
    height: u32,
    num_slots: u32,
    slot_size: usize,
    seq: u64,
}

impl CameraRingWriter {
    pub fn new(path: &Path, width: u32, height: u32, num_slots: u32) -> io::Result<Self> {
        if width == 0 || height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ring needs a positive width/height",
            ));
        }
        let stride = width as usize * 3;
        let slot_size = stride * height as usize;
        let total = HEADER_BYTES + num_slots as usize * slot_size;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o666)
            .open(path)?;
        file.set_len(total as u64)?;

        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        let mut this = Self {
            ptr: addr as *mut u8,
            len: total,
            _file: file,
            width,
            height,
            num_slots,
            slot_size,
            seq: 0,
        };
        this.write_header();
        Ok(this)
    }

    fn write_header(&mut self) {
        let h = unsafe { std::slice::from_raw_parts_mut(self.ptr, HEADER_BYTES) };
        for b in h.iter_mut() {
            *b = 0;
        }
        h[OFF_MAGIC..OFF_MAGIC + 4].copy_from_slice(&MAGIC.to_le_bytes());
        h[OFF_FMT..OFF_FMT + 4].copy_from_slice(&FMT_RGB8.to_le_bytes());
        h[OFF_WIDTH..OFF_WIDTH + 4].copy_from_slice(&self.width.to_le_bytes());
        h[OFF_HEIGHT..OFF_HEIGHT + 4].copy_from_slice(&self.height.to_le_bytes());
        h[OFF_STRIDE..OFF_STRIDE + 4].copy_from_slice(&(self.width * 3).to_le_bytes());
        h[OFF_SLOT_SIZE..OFF_SLOT_SIZE + 4].copy_from_slice(&(self.slot_size as u32).to_le_bytes());
        h[OFF_SLOTS..OFF_SLOTS + 4].copy_from_slice(&self.num_slots.to_le_bytes());
        let pid = std::process::id();
        h[OFF_PID..OFF_PID + 4].copy_from_slice(&pid.to_le_bytes());
        h[OFF_SEQ..OFF_SEQ + 8].copy_from_slice(&0u64.to_le_bytes());
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    pub fn stride(&self) -> usize {
        self.width as usize * 3
    }

    pub fn submit_with(&mut self, write: impl FnOnce(&mut [u8])) {
        self.seq += 1;
        let slot = (self.seq % self.num_slots as u64) as usize;
        let off = HEADER_BYTES + slot * self.slot_size;
        let payload = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), self.slot_size) };
        write(payload);
        compiler_fence(Ordering::Release);
        let seq_bytes = self.seq.to_le_bytes();
        for (i, byte) in seq_bytes.iter().enumerate() {
            unsafe { std::ptr::write_volatile(self.ptr.add(OFF_SEQ + i), *byte) };
        }
    }

    pub fn seq(&self) -> u64 {
        self.seq
    }
}

impl Drop for CameraRingWriter {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}
