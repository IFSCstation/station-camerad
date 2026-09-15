#!/usr/bin/env bash
# Install camerad as a user systemd service (no root required: the user owns
# the V4L2 device via the `video` group and /dev/shm is world-writable).
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
bin_dir="$HOME/.local/bin"
unit_dir="$HOME/.config/systemd/user"
unit="$unit_dir/station-camerad.service"

cargo build --release --manifest-path "$repo/Cargo.toml"

install -Dm755 "$repo/target/release/camerad" "$bin_dir/camerad"
install -Dm644 "$repo/deploy/station-camerad.service" "$unit"

systemctl --user daemon-reload
systemctl --user enable --now station-camerad.service
systemctl --user --no-pager status station-camerad.service