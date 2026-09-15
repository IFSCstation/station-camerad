# station-camerad — implementation plan

Drop-in Rust replacement for `camera_feeder.py` (the sole owner of the webcam
that publishes RGB8 frames to the shared cam_ring).

Status: **DONE** — port implemented, gated clean, verified live against the
existing Python reader. systemd deferred by owner.

## Goal

A `camerad` binary that captures YUYV from `/dev/videoN` via raw V4L2 ioctl +
mmap, converts to RGB8 in one pass (flip folded in), and writes straight into
the cam_ring slot — **no intermediate frame buffer, no OpenCV**. Must stay
byte-compatible with the ring so the existing Godot GDExtension and the Python
tracker read it unchanged.

## Milestones

- [x] Scaffold: `Cargo.toml` (lib `camerad` + bin `camerad`, dep `libc` only), `.gitignore`
- [x] `src/ring.rs` — `CameraRingWriter` (header layout, submit path)
- [x] `src/yuyv.rs` — single-pass YUYV→RGB + mirror
- [x] `src/v4l2.rs` — raw V4L2 capture (ioctl, mmap buffers, blocking DQBUF/QBUF)
- [x] `src/lib.rs` + `src/bin/camerad.rs` — CLI + main loop (no `nix`, `libc::sigaction`)
- [x] `tests/ring.rs` — integration tests (byte-level header/slots/mirror)
- [x] `AGENTS.md` + `README.md`
- [x] Gate clean: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo build --release`
- [x] Live smoke test vs `cam_ring_peek.py` (unchanged consumer)
- [ ] systemd unit (service; socket if we later add a control plane)

## Ring protocol (source of truth: `old_layout_ignore/webcam_test/python/cam_ring.py`)

64-byte header, little-endian `<IIIIIIIIQ` (no padding):

| off | size | field |
|----:|----:|-------|
| 0 | u32 | magic `0xCAC00001` |
| 4 | u32 | fmt = 1 (RGB8) |
| 8 | u32 | width |
| 12 | u32 | height |
| 16 | u32 | stride = width*3 |
| 20 | u32 | slot_size = stride*height |
| 24 | u32 | num_slots = 3 |
| 28 | u32 | writer_pid |
| 32 | u64 | producer_seq |

Payload for frame `seq` at `64 + (seq % 3) * slot_size`. Writer order: payload
then seq (Release fence + volatile store) so consumers re-baseline on PID
change and never see torn headers.

Default path: `/dev/shm/body_estim_camera`.

## V4L2 struct layouts (probed from `/usr/include/linux/videodev2.h`, kernel 7.x)

`v4l2_capability` = 104 · `v4l2_pix_format` = 48 · `v4l2_format` = 208,
`type@0` + **union (pix)@8** (the fmt union is 8-byte aligned — a `u32` pad
sits between `type` and `pix`) · `v4l2_buffer` = 88 (`memory@60`, `m@64`,
`length@72`, `request_fd@80`; `timestamp` is 8-aligned so a pad sits between
`field` and `timestamp`). Compile-time `size_of`/`offset_of` asserts guard
each. IOCTL request codes are computed from the standard `_IOC` macro
(no `nix`): QUERYCAP `_IOR('V',0,…)` · G_FMT n.4 · S_FMT n.5 · REQBUFS n.8 ·
QUERYBUF n.9 · QBUF n.15 · DQBUF n.17 · STREAMON `_IOW('V',18,int)` ·
STREAMOFF `_IOW('V',19,int)`.

## CLI (identical to camera_feeder.py)

`--camera` (0 or device path) · `--path` (ring file) · `--mirror` · `--fps` ·
`--preview` (no window; prints ring geometry + publish stats).

## Verification

1. `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test && cargo build --release`
2. `tests/ring.rs` byte-compat read-back of a written ring.
3. Live: `./target/release/camerad --mirror --fps 30`, then the **existing**
   `cam_ring_peek.py` reads the same ring as if the Python feeder were running.

## Out of scope (later)

MJPEG decode, preview window, Unix-socket JSON control plane (future
`camerad` socket per ifscstation-godot-sdk), `run.sh` swap in the deprecated
`old_layout_ignore` tree, systemd unit (explicitly deferred by owner).