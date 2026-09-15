# station-camerad

Drop-in Rust replacement for `camera_feeder.py`: owns the webcam, captures
YUYV via raw V4L2 ioctl + mmap, and publishes RGB8 frames into the shared
cam_ring at `/dev/shm/body_estim_camera` — byte-compatible with the existing
Godot/Python consumers. One dependency (`libc`), no OpenCV, no intermediate
frame buffer.

## Build & run

```sh
cargo build --release
./target/release/camerad --mirror --fps 30 --camera 0
```

Stop with Ctrl-C (SIGTERM/SIGINT) — the feeder drains the camera and prints
a final frame count.

## CLI (identical to camera_feeder.py)

| flag | default | meaning |
|------|---------|---------|
| `--camera <index\|path>` | `0` | `/dev/videoN` or any V4L2 device path |
| `--path <ring>` | `/dev/shm/body_estim_camera` | shared-memory ring file |
| `--mirror` | off | flip frames horizontally (Godot + tracker see the mirrored feed) |
| `--fps <fps>` | `0` | cap the publish rate (0 = as fast as the camera) |
| `--preview` | — | no window; prints ring geometry + publish stats |
| `-v` / `--log` | off | log the startup line and periodic `published …` stats |

Default run is silent (errors still go to stderr). With `--log`, output matches
the Python feeder format: `[feeder] camera … -> WxH ring=…`,
`[feeder] published N frames (F fps)` every 60 frames,
`[feeder] stopping after N frames`.

## Camera ring

64-byte little-endian header `<IIIIIIIIQ`: magic `0xCAC00001`, fmt=1 (RGB8),
width, height, stride=width*3, slot_size, num_slots=3, writer pid, `u64`
producer_seq. Frame `seq` lives at `64 + (seq % 3) * slot_size`; the writer
stores the payload first, then publishes `seq` (Release fence + volatile
store). Protocol source of truth:
`old_layout_ignore/webcam_test/python/cam_ring.py`.

## Verification

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
```

Live smoke test (device `/dev/video0` optional — on the dev host it's the
DroidCam loopback):

```sh
./target/release/camerad --mirror --fps 30 --camera 0 &
old_layout_ignore/webcam_test/python/cam_ring_peek.py --seconds 3 --save /tmp/frame.png
```

## Notes

- V4L2 ABI is hand-declared (`#[repr(C)]`) with compile-time `size_of`/
  `offset_of` asserts; device layouts verified on kernel 7.x.
- YUYV-only, BT.601 limited-range fixed-point → RGB; mirror folded into the
  conversion.
- The old Python feeder lives in the deprecated `old_layout_ignore` tree; this
  repo is its replacement (systemd unit is deferred).