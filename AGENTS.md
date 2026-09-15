# AGENTS.md

Guidance for AI agents / humans working in this repo.

## What this is

`station-camerad` is a drop-in Rust replacement for
`old_layout_ignore/webcam_test/python/camera_feeder.py`. It is the **sole owner
of the webcam**: it captures YUYV via raw V4L2 ioctl + mmap, converts to RGB8
in one pass (mirror folded in), and publishes into the shared camera ring at
`/dev/shm/body_estim_camera`.

The ring protocol — 64-byte LE header `<IIIIIIIIQ`, magic `0xCAC00001`,
fmt=1 (RGB8), payload at `64 + (seq % 3) * slot_size` — **must stay
byte-compatible**. Source of truth:
`old_layout_ignore/webcam_test/python/cam_ring.py`. Consumers (Godot
GDExtension, Python tracker) read it unchanged; changing the layout silently
breaks them.

## Layout

- `src/ring.rs` — ring writer (header, `submit_with`, mmap, seq publish)
- `src/yuyv.rs` — single-pass YUYV→RGB8 + mirror, pure functions + unit tests
- `src/v4l2.rs` — raw capture: hand-rolled V4L2 ABI, blocking `dequeue`,
  `frame`/`release`, `Camera::open`
- `src/bin/camerad.rs` — CLI + signal handling (`libc::sigaction`, no SA_RESTART
  so blocking DQBUF sees EINTR) + main loop
- `tests/ring.rs` — integration tests; **never use the live camera in tests**

## Hard rules

1. **Single dependency.** Only `libc` in `Cargo.toml`. No `nix`, no OpenCV,
   no bindgen-generated structs for V4L2.
2. **No kernel interference.** Userspace V4L2 client only: ioctls on the
   device fd + mmap. No module loading, sysfs writes, or v4l2loopback setup.
3. **YUYV only.** The port targets loopback + droidcam devices which are
   YUYV-only. Do not add MJPEG.
4. **Keep the ring byte-compatible.** Header offsets, magic, ordering
   (payload then seq after a Release fence + volatile store) never change.
5. **Pass the gates before finishing.**
   `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test && cargo build --release`
6. V4L2 ABI structs are hand-declared `#[repr(C)]` with compile-time
   `size_of`/`offset_of` asserts in `src/v4l2.rs`; `offset_of!` needs a reason
   to trigger — device layouts observed on kernel 7.x put the `v4l2_format`
   union at offset 8 and `v4l2_buffer.memory` at offset 60. Keep those asserts
   if you touch the structs.
7. No comments in code unless asked.

## Environment

- Host: Arch Linux, kernel 7.x. `/dev/video0` = v4l2loopback "DroidCam Virtual
  Camera" (YUYV 1280x720, fed by a running `droidcam` process; no physical
  webcam needed).
- Testing against the live device is optional (device may be absent): run
  `./target/release/camerad --mirror --fps 30 --camera 0` and read the ring
  with `cam_ring_peek.py` in the `old_layout_ignore` tree.
- Native build on the Pi (arm64) must also pass — the ABI structs are portable
  (little-endian, 64-bit).