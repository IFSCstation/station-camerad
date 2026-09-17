const SCALE: i32 = 1024;
const K_Y: i32 = 1192;
const K_V_R: i32 = 1634;
const K_V_G: i32 = 833;
const K_U_G: i32 = 400;
const K_U_B: i32 = 2066;

fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn convert(y: i32, u: i32, v: i32) -> [u8; 3] {
    let base = K_Y * (y - 16);
    let r = (base + K_V_R * v) / SCALE;
    let g = (base - K_V_G * v - K_U_G * u) / SCALE;
    let b = (base + K_U_B * u) / SCALE;
    [clamp8(r), clamp8(g), clamp8(b)]
}

fn yuyv_to_rgb8_scalar(src: &[u8], dst: &mut [u8], width: usize, height: usize, mirror: bool) {
    let mut s = 0;
    for y in 0..height {
        let row_base = y * width;
        let mut x = 0;
        while x < width {
            let u = src[s + 1] as i32 - 128;
            let v = src[s + 3] as i32 - 128;
            let y0 = src[s] as i32;
            let y1 = src[s + 2] as i32;
            let a = convert(y0, u, v);
            let b = convert(y1, u, v);
            for (xv, rgb) in [(x, a), (x + 1, b)] {
                let dx = if mirror { width - 1 - xv } else { xv };
                let o = (row_base + dx) * 3;
                dst[o] = rgb[0];
                dst[o + 1] = rgb[1];
                dst[o + 2] = rgb[2];
            }
            s += 4;
            x += 2;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{K_U_B, K_U_G, K_V_G, K_V_R, K_Y};
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn quantize(a: int32x4_t) -> uint8x8_t {
        let bias = vandq_s32(vshrq_n_s32(a, 31), vdupq_n_s32(1023));
        let q = vshrq_n_s32(vaddq_s32(a, bias), 10);
        let t16 = vqmovun_s32(q);
        vqmovn_u16(vcombine_u16(t16, vdup_n_u16(0)))
    }

    #[inline(always)]
    unsafe fn yuyv_4px(s: *const u8, d: *mut u8, rev: bool) {
        let b = vld1_u8(s);
        let uzp = vuzp_u8(b, b);
        let y8 = uzp.0;
        let c8 = uzp.1;

        let uv = vuzp_u8(c8, c8);
        let u_dup = vzip_u8(uv.0, uv.0).0;
        let v_dup = vzip_u8(uv.1, uv.1).0;

        let y_s16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(y8)), vdupq_n_s16(16));
        let y32 = vmovl_s16(vget_low_s16(y_s16));

        let u_s16 = vreinterpretq_s16_u16(vmovl_u8(u_dup));
        let u32 = vsubq_s32(vmovl_s16(vget_low_s16(u_s16)), vdupq_n_s32(128));

        let v_s16 = vreinterpretq_s16_u16(vmovl_u8(v_dup));
        let v32 = vsubq_s32(vmovl_s16(vget_low_s16(v_s16)), vdupq_n_s32(128));

        let base = vmulq_s32(y32, vdupq_n_s32(K_Y));
        let r = quantize(vmlaq_n_s32(base, v32, K_V_R));
        let g = quantize(vmlsq_n_s32(vmlsq_n_s32(base, v32, K_V_G), u32, K_U_G));
        let b_out = quantize(vmlaq_n_s32(base, u32, K_U_B));

        if rev {
            *d = vget_lane_u8(r, 3);
            *d.add(1) = vget_lane_u8(g, 3);
            *d.add(2) = vget_lane_u8(b_out, 3);
            *d.add(3) = vget_lane_u8(r, 2);
            *d.add(4) = vget_lane_u8(g, 2);
            *d.add(5) = vget_lane_u8(b_out, 2);
            *d.add(6) = vget_lane_u8(r, 1);
            *d.add(7) = vget_lane_u8(g, 1);
            *d.add(8) = vget_lane_u8(b_out, 1);
            *d.add(9) = vget_lane_u8(r, 0);
            *d.add(10) = vget_lane_u8(g, 0);
            *d.add(11) = vget_lane_u8(b_out, 0);
        } else {
            *d = vget_lane_u8(r, 0);
            *d.add(1) = vget_lane_u8(g, 0);
            *d.add(2) = vget_lane_u8(b_out, 0);
            *d.add(3) = vget_lane_u8(r, 1);
            *d.add(4) = vget_lane_u8(g, 1);
            *d.add(5) = vget_lane_u8(b_out, 1);
            *d.add(6) = vget_lane_u8(r, 2);
            *d.add(7) = vget_lane_u8(g, 2);
            *d.add(8) = vget_lane_u8(b_out, 2);
            *d.add(9) = vget_lane_u8(r, 3);
            *d.add(10) = vget_lane_u8(g, 3);
            *d.add(11) = vget_lane_u8(b_out, 3);
        }
    }

    pub unsafe fn yuyv_to_rgb8_neon(
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        height: usize,
        mirror: bool,
    ) {
        let groups = width / 4;
        for y in 0..height {
            let row_in = &src[y * width * 2..][..width * 2];
            let row_out = &mut dst[y * width * 3..][..width * 3];
            for g in 0..groups {
                let si = g * 8;
                let di = if mirror {
                    (groups - 1 - g) * 12
                } else {
                    g * 12
                };
                yuyv_4px(
                    row_in.as_ptr().add(si),
                    row_out.as_mut_ptr().add(di),
                    mirror,
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn fast_path_available() -> bool {
    std::env::var_os("CAMERAD_FORCE_SCALAR").is_none()
        && std::arch::is_aarch64_feature_detected!("neon")
}

pub fn rgb24_to_rgb8(src: &[u8], dst: &mut [u8], width: usize, height: usize, mirror: bool) {
    assert!(src.len() >= width * height * 3);
    assert!(dst.len() >= width * height * 3);
    if !mirror {
        dst[..src.len()].copy_from_slice(src);
        return;
    }
    for y in 0..height {
        let row_in = &src[y * width * 3..][..width * 3];
        let row_out = &mut dst[y * width * 3..][..width * 3];
        for x in 0..width {
            let si = x * 3;
            let di = (width - 1 - x) * 3;
            row_out[di] = row_in[si];
            row_out[di + 1] = row_in[si + 1];
            row_out[di + 2] = row_in[si + 2];
        }
    }
}

pub fn yuyv_to_rgb8(src: &[u8], dst: &mut [u8], width: usize, height: usize, mirror: bool) {
    assert_eq!(width % 2, 0);
    assert!(src.len() >= width * height * 2);
    assert!(dst.len() >= width * height * 3);

    #[cfg(target_arch = "aarch64")]
    if width >= 4 && width % 4 == 0 && fast_path_available() {
        unsafe {
            neon::yuyv_to_rgb8_neon(src, dst, width, height, mirror);
        }
        return;
    }
    yuyv_to_rgb8_scalar(src, dst, width, height, mirror);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb24_passthrough_no_mirror() {
        let src: Vec<u8> = (0..24).collect();
        let mut dst = vec![0u8; 24];
        rgb24_to_rgb8(&src, &mut dst, 2, 1, false);
        assert_eq!(dst, src);
    }

    #[test]
    fn rgb24_mirror_reverses_row() {
        let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut dst = vec![0u8; 12];
        rgb24_to_rgb8(&src, &mut dst, 4, 1, true);
        assert_eq!(&dst[0..3], &[10, 11, 12]);
        assert_eq!(&dst[3..6], &[7, 8, 9]);
        assert_eq!(&dst[6..9], &[4, 5, 6]);
        assert_eq!(&dst[9..12], &[1, 2, 3]);
    }

    #[test]
    fn rgb24_mirror_multi_row() {
        let src: Vec<u8> = (0..24).collect();
        let mut dst = vec![0u8; 24];
        rgb24_to_rgb8(&src, &mut dst, 2, 2, true);
        assert_eq!(&dst[0..3], &[3, 4, 5]);
        assert_eq!(&dst[3..6], &[0, 1, 2]);
        assert_eq!(&dst[6..9], &[9, 10, 11]);
        assert_eq!(&dst[9..12], &[6, 7, 8]);
    }

    #[test]
    fn gray_is_gray() {
        let src = [128u8; 8];
        let mut dst = [0u8; 6];
        yuyv_to_rgb8(&src, &mut dst, 2, 1, false);
        assert_eq!(dst, [130, 130, 130, 130, 130, 130]);
    }

    #[test]
    fn mirror_reverses_row() {
        let mut src = vec![0u8; 16];
        src[0] = 82;
        src[1] = 90;
        src[2] = 30;
        src[3] = 240;
        src[4] = 200;
        src[5] = 160;
        src[6] = 150;
        src[7] = 240;
        src[8] = 70;
        src[9] = 90;
        src[10] = 40;
        src[11] = 240;
        src[12] = 220;
        src[13] = 160;
        src[14] = 210;
        src[15] = 240;
        let mut plain = vec![0u8; 12];
        let mut folded = vec![0u8; 12];
        yuyv_to_rgb8(&src, &mut plain, 4, 1, false);
        yuyv_to_rgb8(&src, &mut folded, 4, 1, true);
        assert_eq!(folded[0..3], plain[9..12]);
        assert_eq!(folded[3..6], plain[6..9]);
        assert_eq!(folded[6..9], plain[3..6]);
        assert_eq!(folded[9..12], plain[0..3]);
    }

    #[test]
    fn blue_dominates_for_high_u() {
        let src = [82, 200, 150, 90];
        let mut dst = [0u8; 6];
        yuyv_to_rgb8(&src, &mut dst, 2, 1, false);
        assert!(dst[2] > dst[0], "expected blue-ish, got {dst:?}");
        assert!(dst[5] > dst[3], "expected blue-ish, got {dst:?}");
    }

    #[test]
    fn mirror_reverses_row_eight() {
        let mut rng = 0x1234_5678_9abc_def0u64;
        let next = |rng: &mut u64| {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        };
        for _ in 0..64 {
            let mut src = vec![0u8; 8 * 2];
            for b in src.iter_mut() {
                next(&mut rng);
                *b = (rng >> 56) as u8;
            }
            let mut plain = vec![0u8; 8 * 3];
            let mut folded = vec![0u8; 8 * 3];
            yuyv_to_rgb8(&src, &mut plain, 8, 1, false);
            yuyv_to_rgb8(&src, &mut folded, 8, 1, true);
            for i in 0..8 {
                assert_eq!(
                    &folded[i * 3..i * 3 + 3],
                    &plain[(7 - i) * 3..(7 - i) * 3 + 3],
                    "mirror split mismatch at pixel {i} for src {src:?}"
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_byte_exact() {
        let mut rng = 0xdead_beef_cafe_babeu64;
        let next = |rng: &mut u64| {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        };
        let mut mismatches = 0usize;
        for trial in 0..64 {
            let mut src = vec![0u8; 16 * 2 * 4];
            for b in src.iter_mut() {
                next(&mut rng);
                *b = if trial < 16 {
                    (rng >> 56) as u8
                } else {
                    (rng % 179) as u8
                };
            }
            for mirror in [false, true] {
                for w in [8usize, 16] {
                    let h = 4usize;
                    let mut fast = vec![0u8; w * h * 3];
                    let mut scalar = vec![0u8; w * h * 3];
                    let src_w = &src[..w * h * 2];
                    yuyv_to_rgb8(src_w, &mut fast, w, h, mirror);
                    yuyv_to_rgb8_scalar(src_w, &mut scalar, w, h, mirror);
                    for i in 0..fast.len() {
                        if fast[i] != scalar[i] {
                            mismatches += 1;
                            if mismatches <= 5 {
                                let px = i / 3;
                                let ch = i % 3;
                                panic!(
                                    "mismatch trial={trial} mirror={mirror} w={w} px={px} ch={ch}: fast={} scalar={}",
                                    fast[i], scalar[i]
                                );
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "neon/scalar diverged on {mismatches}/{} bytes",
            9 * 64
        );
    }
}
