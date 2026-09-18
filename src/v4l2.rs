use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::raw::c_int;
use std::path::Path;

use libc::{MAP_FAILED, MAP_SHARED, PROT_READ, PROT_WRITE};

const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const V4L2_MEMORY_MMAP: u32 = 1;
const V4L2_FIELD_ANY: u32 = 0;
pub const V4L2_PIX_FMT_YUYV: u32 = 0x5659_5559;
pub const V4L2_PIX_FMT_RGB24: u32 = 0x3342_4752;
pub const V4L2_PIX_FMT_YUV420: u32 = 0x3231_5559;
const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
const V4L2_CAP_STREAMING: u32 = 0x0400_0000;
const NUM_BUFFERS: u32 = 4;

const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = 8;
const IOC_SIZESHIFT: u32 = 16;
const IOC_DIRSHIFT: u32 = 30;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const IOC_READWRITE: u32 = 3;

const fn ioc(dir: u32, ty: u8, nr: u8, size: u32) -> u32 {
    (dir << IOC_DIRSHIFT)
        | ((ty as u32) << IOC_TYPESHIFT)
        | ((nr as u32) << IOC_NRSHIFT)
        | (size << IOC_SIZESHIFT)
}

const fn ior(ty: u8, nr: u8, size: u32) -> u32 {
    ioc(IOC_READ, ty, nr, size)
}

const fn iowr(ty: u8, nr: u8, size: u32) -> u32 {
    ioc(IOC_READWRITE, ty, nr, size)
}

const fn iow_int(ty: u8, nr: u8) -> u32 {
    ioc(IOC_WRITE, ty, nr, std::mem::size_of::<c_int>() as u32)
}

const VIDIOC_QUERYCAP: u32 = ior(b'V', 0, 104);
const VIDIOC_G_FMT: u32 = iowr(b'V', 4, 208);
const VIDIOC_S_FMT: u32 = iowr(b'V', 5, 208);
const VIDIOC_REQBUFS: u32 = iowr(b'V', 8, 20);
const VIDIOC_QUERYBUF: u32 = iowr(b'V', 9, 88);
const VIDIOC_QBUF: u32 = iowr(b'V', 15, 88);
const VIDIOC_DQBUF: u32 = iowr(b'V', 17, 88);
const VIDIOC_STREAMON: u32 = iow_int(b'V', 18);
const VIDIOC_STREAMOFF: u32 = iow_int(b'V', 19);

#[allow(dead_code)]
#[repr(C)]
struct V4l2Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

#[allow(dead_code)]
#[repr(C)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

#[allow(dead_code)]
#[repr(C)]
struct V4l2Format {
    type_: u32,
    _pad: u32,
    pix: V4l2PixFormat,
    _rest: [u8; 152],
}

const _: () = assert!(std::mem::size_of::<V4l2Format>() == 208);
const _: () = assert!(std::mem::offset_of!(V4l2Format, pix) == 8);

#[allow(dead_code)]
#[repr(C)]
struct V4l2RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    reserved: [u32; 2],
}

#[allow(dead_code)]
#[repr(C)]
union V4l2BufferM {
    offset: u32,
    userptr: usize,
    planes: *mut libc::c_void,
    fd: i32,
}

#[allow(dead_code)]
#[repr(C)]
struct V4l2Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    _pad: [u8; 4],
    timestamp: [u8; 16],
    timecode: [u8; 16],
    sequence: u32,
    memory: u32,
    m: V4l2BufferM,
    length: u32,
    reserved2: u32,
    request_fd: i32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<V4l2Capability>() == 104);
const _: () = assert!(std::mem::size_of::<V4l2PixFormat>() == 48);
const _: () = assert!(std::mem::size_of::<V4l2Format>() == 208);
const _: () = assert!(std::mem::size_of::<V4l2RequestBuffers>() == 20);
const _: () = assert!(std::mem::size_of::<V4l2Buffer>() == 88);
const _: () = assert!(std::mem::offset_of!(V4l2Buffer, memory) == 60);

unsafe fn ioctl_ret(fd: c_int, req: u32, arg: *mut libc::c_void) -> io::Result<c_int> {
    let r = unsafe { libc::ioctl(fd, req as libc::c_ulong, arg) };
    if r < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(r)
    }
}

macro_rules! ioctl {
    ($fd:expr, $req:expr, $arg:expr) => {
        unsafe { ioctl_ret($fd, $req as u32, $arg as *const _ as *mut libc::c_void)? }
    };
}

fn unsupported(msg: impl Into<String>) -> io::Error {
    io::Error::other(msg.into())
}

struct MappedBuffer {
    addr: *mut u8,
    len: usize,
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.addr as *mut libc::c_void, self.len);
        }
    }
}

pub struct Camera {
    file: OwnedFd,
    buffers: Vec<MappedBuffer>,
    width: u32,
    height: u32,
    pixel_format: u32,
    last_index: usize,
    last_bytesused: usize,
}

impl Camera {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
        {
            Ok(f) => f,
            Err(_) => std::fs::OpenOptions::new().read(true).open(path)?,
        };
        let fd = OwnedFd::from(file);

        let mut cap: V4l2Capability = unsafe { std::mem::zeroed() };
        ioctl!(fd.as_raw_fd(), VIDIOC_QUERYCAP, &mut cap);
        let caps = if cap.device_caps != 0 {
            cap.device_caps
        } else {
            cap.capabilities
        };
        if caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING)
            != (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING)
        {
            return Err(unsupported(format!(
                "{}: not a streaming video-capture device",
                path.display()
            )));
        }

        let (width, height, pixel_format) = Self::negotiate_format(&fd)?;
        let buffers = Self::setup_buffers(&fd)?;

        let mut camera = Camera {
            file: fd,
            buffers,
            width,
            height,
            pixel_format,
            last_index: 0,
            last_bytesused: 0,
        };
        camera.stream_on()?;
        Ok(camera)
    }

    fn negotiate_format(fd: &OwnedFd) -> io::Result<(u32, u32, u32)> {
        let mut fmt: V4l2Format = unsafe { std::mem::zeroed() };
        fmt.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl!(fd.as_raw_fd(), VIDIOC_G_FMT, &mut fmt);
        Self::try_set(fd, V4L2_PIX_FMT_RGB24);
        let mut gf: V4l2Format = unsafe { std::mem::zeroed() };
        gf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl!(fd.as_raw_fd(), VIDIOC_G_FMT, &mut gf);
        if gf.pix.pixelformat == V4L2_PIX_FMT_RGB24 && gf.pix.width != 0 && gf.pix.height != 0 {
            return Ok((gf.pix.width, gf.pix.height, V4L2_PIX_FMT_RGB24));
        }
        Self::try_set(fd, V4L2_PIX_FMT_YUYV);
        let mut gy: V4l2Format = unsafe { std::mem::zeroed() };
        gy.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl!(fd.as_raw_fd(), VIDIOC_G_FMT, &mut gy);
        if gy.pix.pixelformat == V4L2_PIX_FMT_YUYV && gy.pix.width != 0 && gy.pix.height != 0 {
            return Ok((gy.pix.width, gy.pix.height, V4L2_PIX_FMT_YUYV));
        }
        Self::try_set(fd, V4L2_PIX_FMT_YUV420);
        let mut gi: V4l2Format = unsafe { std::mem::zeroed() };
        gi.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl!(fd.as_raw_fd(), VIDIOC_G_FMT, &mut gi);
        if gi.pix.pixelformat != V4L2_PIX_FMT_YUV420 {
            return Err(unsupported("camera provides neither RGB24, YUYV, nor YU12"));
        }
        if gi.pix.width == 0 || gi.pix.height == 0 {
            return Err(unsupported("camera reported a zero-size frame"));
        }
        Ok((gi.pix.width, gi.pix.height, V4L2_PIX_FMT_YUV420))
    }

    fn try_set(fd: &OwnedFd, fmtc: u32) {
        let mut nf: V4l2Format = unsafe { std::mem::zeroed() };
        nf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        let mut gf: V4l2Format = unsafe { std::mem::zeroed() };
        gf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        let _ = unsafe {
            ioctl_ret(
                fd.as_raw_fd(),
                VIDIOC_G_FMT,
                &mut gf as *mut _ as *mut libc::c_void,
            )
        };
        nf.pix.width = if gf.pix.width == 0 { 640 } else { gf.pix.width };
        nf.pix.height = if gf.pix.height == 0 {
            480
        } else {
            gf.pix.height
        };
        nf.pix.pixelformat = fmtc;
        nf.pix.field = V4L2_FIELD_ANY;
        let _ = unsafe {
            ioctl_ret(
                fd.as_raw_fd(),
                VIDIOC_S_FMT,
                &mut nf as *mut _ as *mut libc::c_void,
            )
        };
    }

    fn setup_buffers(fd: &OwnedFd) -> io::Result<Vec<MappedBuffer>> {
        let mut rb: V4l2RequestBuffers = unsafe { std::mem::zeroed() };
        rb.count = NUM_BUFFERS;
        rb.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        rb.memory = V4L2_MEMORY_MMAP;
        ioctl!(fd.as_raw_fd(), VIDIOC_REQBUFS, &mut rb);
        if rb.count == 0 {
            return Err(unsupported("camera refused mmap buffers"));
        }
        let mut buffers = Vec::with_capacity(rb.count as usize);
        for i in 0..rb.count {
            let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
            buf.index = i;
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            ioctl!(fd.as_raw_fd(), VIDIOC_QUERYBUF, &mut buf);
            let len = buf.length as usize;
            let offset = unsafe { buf.m.offset } as usize;
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    fd.as_raw_fd(),
                    offset as libc::off_t,
                )
            };
            if addr == MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            buffers.push(MappedBuffer {
                addr: addr as *mut u8,
                len,
            });
        }
        for i in 0..rb.count {
            let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
            buf.index = i;
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            ioctl!(fd.as_raw_fd(), VIDIOC_QBUF, &mut buf);
        }
        Ok(buffers)
    }

    fn stream_on(&mut self) -> io::Result<()> {
        let t = V4L2_BUF_TYPE_VIDEO_CAPTURE as c_int;
        ioctl!(self.file.as_raw_fd(), VIDIOC_STREAMON, &t);
        Ok(())
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixel_format(&self) -> u32 {
        self.pixel_format
    }

    pub fn dequeue(&mut self) -> io::Result<()> {
        let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
        buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        ioctl!(self.file.as_raw_fd(), VIDIOC_DQBUF, &mut buf);
        self.last_index = buf.index as usize;
        self.last_bytesused = buf.bytesused as usize;
        Ok(())
    }

    pub fn frame(&self) -> &[u8] {
        let buf = &self.buffers[self.last_index];
        let len = self.last_bytesused.min(buf.len);
        unsafe { std::slice::from_raw_parts(buf.addr, len) }
    }

    pub fn release(&mut self) -> io::Result<()> {
        let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
        buf.index = self.last_index as u32;
        buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        ioctl!(self.file.as_raw_fd(), VIDIOC_QBUF, &mut buf);
        Ok(())
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        let t = V4L2_BUF_TYPE_VIDEO_CAPTURE as c_int;
        let _ =
            unsafe { libc::ioctl(self.file.as_raw_fd(), VIDIOC_STREAMOFF as libc::c_ulong, &t) };
    }
}
