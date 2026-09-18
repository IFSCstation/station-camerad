use std::env;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use camerad::ring::{CameraRingWriter, CAM_RING_DEFAULT};
use camerad::v4l2::{Camera, V4L2_PIX_FMT_RGB24, V4L2_PIX_FMT_YUV420};
use camerad::yuyv;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

struct Args {
    camera: String,
    path: String,
    fps: f64,
    preview: bool,
    log: bool,
}

fn usage() -> String {
    format!(
        "usage: camerad [--camera <index|path>] [--path <ring>] [--fps <fps>] [--preview] [-v|--log]\n\
         owns the webcam, publishes RGB8 frames to the shared cam_ring\n\
         --camera  camera index or device path (default 0)\n\
         --path    shared-memory ring file (default {CAM_RING_DEFAULT})\n\
         --fps     cap the publish rate (0 = as fast as the camera)\n\
         --preview no window; prints ring geometry + publish statistics\n\
         -v, --log log the startup line and publish statistics (default: silent)"
    )
}

fn parse_args() -> std::result::Result<Args, String> {
    let mut args = Args {
        camera: "0".into(),
        path: CAM_RING_DEFAULT.into(),
        fps: 0.0,
        preview: false,
        log: false,
    };
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--camera" => args.camera = it.next().ok_or("--camera needs a value")?,
            "--path" => args.path = it.next().ok_or("--path needs a value")?,
            "--preview" => args.preview = true,
            "-v" | "--log" => args.log = true,
            "--fps" => {
                let v = it.next().ok_or("--fps needs a value")?;
                args.fps = v.parse().map_err(|_| format!("invalid fps {v:?}"))?;
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }
    Ok(args)
}

fn resolve_device(spec: &str) -> PathBuf {
    if !spec.is_empty() && spec.chars().all(|c| c.is_ascii_digit()) {
        PathBuf::from(format!("/dev/video{spec}"))
    } else {
        PathBuf::from(spec)
    }
}

fn setup_signal_handlers() -> io::Result<()> {
    let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
    sa.sa_sigaction = on_signal as *const () as libc::sighandler_t;
    unsafe { libc::sigemptyset(&mut sa.sa_mask) };
    sa.sa_flags = 0;
    for sig in [libc::SIGINT, libc::SIGTERM] {
        if unsafe { libc::sigaction(sig, &sa, std::ptr::null_mut()) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[feeder] {e}");
            return ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[feeder] {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> io::Result<()> {
    setup_signal_handlers()?;

    let device = resolve_device(&args.camera);
    let mut camera = Camera::open(&device)
        .map_err(|e| io::Error::other(format!("could not open camera {:?}: {e}", args.camera)))?;
    let (fw, fh) = (camera.width(), camera.height());
    let pixfmt = camera.pixel_format();
    let is_rgb24 = pixfmt == V4L2_PIX_FMT_RGB24;
    let is_yu12 = pixfmt == V4L2_PIX_FMT_YUV420;
    if args.log {
        let (fmt_tag, mode) = if is_rgb24 {
            ("RGB24", "passthrough")
        } else if is_yu12 {
            ("YU12", "YU12→RGB8")
        } else {
            ("YUYV", "YUYV→RGB8")
        };
        println!(
            "[feeder] camera {} -> {}x{} {fmt_tag} ({mode})  ring={}",
            args.camera, fw, fh, args.path
        );
    }
    if args.preview {
        println!(
            "[feeder] publishing to ring  {}x{}  (preview window disabled in this port)",
            fw, fh
        );
    }

    let mut ring = CameraRingWriter::new(std::path::Path::new(&args.path), fw, fh, 3)?;

    let frame_period = if args.fps > 0.0 { 1.0 / args.fps } else { 0.0 };
    let mut last = Instant::now();
    let mut n: u64 = 0;
    let mut batch_start = Instant::now();

    while !STOP.load(Ordering::SeqCst) {
        if frame_period > 0.0 {
            let dt = last.elapsed().as_secs_f64();
            if dt < frame_period {
                thread::sleep(Duration::from_secs_f64(frame_period - dt));
                continue;
            }
            last = Instant::now();
        }

        if let Err(e) = camera.dequeue() {
            if STOP.load(Ordering::SeqCst) {
                break;
            }
            eprintln!("[feeder] camera read failed ({e}); retrying...");
            thread::sleep(Duration::from_millis(100));
            continue;
        }

        ring.submit_with(|dst| {
            if is_rgb24 {
                yuyv::rgb24_to_rgb8(camera.frame(), dst, fw as usize, fh as usize);
            } else if is_yu12 {
                yuyv::yuv420_to_rgb8(camera.frame(), dst, fw as usize, fh as usize);
            } else {
                yuyv::yuyv_to_rgb8(camera.frame(), dst, fw as usize, fh as usize);
            }
        });
        camera.release()?;

        n += 1;
        if args.log && n.is_multiple_of(60) {
            let elapsed = batch_start.elapsed().as_secs_f64();
            println!(
                "[feeder] published {} frames  ({:.0} fps)",
                n,
                60.0 / elapsed.max(1e-6)
            );
            batch_start = Instant::now();
        }
    }

    if args.log {
        println!("[feeder] stopping after {} frames", n);
    }
    Ok(())
}
