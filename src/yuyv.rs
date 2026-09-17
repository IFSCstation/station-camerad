const K_Y: i32 = 1192;
const K_V_R: i32 = 1634;
const K_V_G: i32 = 833;
const K_U_G: i32 = 400;
const K_U_B: i32 = 2066;

fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn div_scale(v: i32) -> i32 {
    let neg = v < 0;
    let abs = if neg { -v } else { v };
    let q = (abs as u32 >> 10) as i32;
    if neg {
        -q
    } else {
        q
    }
}

fn convert(y: i32, u: i32, v: i32) -> [u8; 3] {
    let base = K_Y * (y - 16);
    let r = div_scale(base + K_V_R * v);
    let g = div_scale(base - K_V_G * v - K_U_G * u);
    let b = div_scale(base + K_U_B * u);
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
    use super::{clamp8, div_scale, K_U_B, K_U_G, K_V_G, K_V_R, K_Y};
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

    pub unsafe fn yuv420_to_rgb8_neon(
        y_plane: *const u8,
        u_plane: *const u8,
        v_plane: *const u8,
        dst: *mut u8,
        width: usize,
        height: usize,
        mirror: bool,
    ) {
        let pixel_groups = width / 8;
        let remainder = width % 8;
        for row in 0..height {
            let y_row = y_plane.add(row * width);
            let uv_row = row / 2;
            let u_row = u_plane.add(uv_row * (width / 2));
            let v_row = v_plane.add(uv_row * (width / 2));
            let row_out = dst.add(row * width * 3);

            for g in 0..pixel_groups {
                let yi = g * 8;
                let ui = g * 4;
                let di = if mirror {
                    (pixel_groups - 1 - g) * 24
                } else {
                    g * 24
                };

                let y8 = vld1q_u8(y_row.add(yi));
                let u4 = vld1_u8(u_row.add(ui));
                let v4 = vld1_u8(v_row.add(ui));

                let u8_lo = vcombine_u8(u4, u4);
                let u8_all = vuzp1q_u8(u8_lo, u8_lo);
                let v8_lo = vcombine_u8(v4, v4);
                let v8_all = vuzp1q_u8(v8_lo, v8_lo);

                let y_s16 =
                    vreinterpretq_s16_u16(vsubq_u16(vmovl_u8(vget_low_u8(y8)), vdupq_n_u16(16)));
                let u_s16 =
                    vreinterpretq_s16_u16(vsubq_u16(vmovl_u8(vget_low_u8(u8_all)), vdupq_n_u16(128)));
                let v_s16 =
                    vreinterpretq_s16_u16(vsubq_u16(vmovl_u8(vget_low_u8(v8_all)), vdupq_n_u16(128)));

                let y_lo = vmovl_s16(vget_low_s16(y_s16));
                let y_hi = vmovl_s16(vget_high_s16(y_s16));
                let u_lo = vmovl_s16(vget_low_s16(u_s16));
                let u_hi = vmovl_s16(vget_high_s16(u_s16));
                let v_lo = vmovl_s16(vget_low_s16(v_s16));
                let v_hi = vmovl_s16(vget_high_s16(v_s16));

                let base_lo = vmulq_s32(y_lo, vdupq_n_s32(K_Y));
                let base_hi = vmulq_s32(y_hi, vdupq_n_s32(K_Y));

                let r_lo = quantize(vmlaq_n_s32(base_lo, v_lo, K_V_R));
                let r_hi = quantize(vmlaq_n_s32(base_hi, v_hi, K_V_R));
                let g_lo = quantize(vmlsq_n_s32(vmlsq_n_s32(base_lo, v_lo, K_V_G), u_lo, K_U_G));
                let g_hi = quantize(vmlsq_n_s32(vmlsq_n_s32(base_hi, v_hi, K_V_G), u_hi, K_U_G));
                let b_lo = quantize(vmlaq_n_s32(base_lo, u_lo, K_U_B));
                let b_hi = quantize(vmlaq_n_s32(base_hi, u_hi, K_U_B));

                let r8 = vcombine_u8(r_lo, r_hi);
                let g8 = vcombine_u8(g_lo, g_hi);
                let b8 = vcombine_u8(b_lo, b_hi);

                let (r8, g8, b8) = if mirror {
                    let r8 = vrev64q_u8(r8);
                    let r8 = vrev32q_u8(r8);
                    let g8 = vrev64q_u8(g8);
                    let g8 = vrev32q_u8(g8);
                    let b8 = vrev64q_u8(b8);
                    let b8 = vrev32q_u8(b8);
                    (r8, g8, b8)
                } else {
                    (r8, g8, b8)
                };

                let o0 = vgetq_lane_u8(r8, 0);
                let o1 = vgetq_lane_u8(g8, 0);
                let o2 = vgetq_lane_u8(b8, 0);
                let o3 = vgetq_lane_u8(r8, 1);
                let o4 = vgetq_lane_u8(g8, 1);
                let o5 = vgetq_lane_u8(b8, 1);
                *row_out.add(di) = o0;
                *row_out.add(di + 1) = o1;
                *row_out.add(di + 2) = o2;
                *row_out.add(di + 3) = o3;
                *row_out.add(di + 4) = o4;
                *row_out.add(di + 5) = o5;
                let o6 = vgetq_lane_u8(r8, 2);
                let o7 = vgetq_lane_u8(g8, 2);
                let o8 = vgetq_lane_u8(b8, 2);
                let o9 = vgetq_lane_u8(r8, 3);
                let o10 = vgetq_lane_u8(g8, 3);
                let o11 = vgetq_lane_u8(b8, 3);
                *row_out.add(di + 6) = o6;
                *row_out.add(di + 7) = o7;
                *row_out.add(di + 8) = o8;
                *row_out.add(di + 9) = o9;
                *row_out.add(di + 10) = o10;
                *row_out.add(di + 11) = o11;
                let o12 = vgetq_lane_u8(r8, 4);
                let o13 = vgetq_lane_u8(g8, 4);
                let o14 = vgetq_lane_u8(b8, 4);
                let o15 = vgetq_lane_u8(r8, 5);
                let o16 = vgetq_lane_u8(g8, 5);
                let o17 = vgetq_lane_u8(b8, 5);
                *row_out.add(di + 12) = o12;
                *row_out.add(di + 13) = o13;
                *row_out.add(di + 14) = o14;
                *row_out.add(di + 15) = o15;
                *row_out.add(di + 16) = o16;
                *row_out.add(di + 17) = o17;
                let o18 = vgetq_lane_u8(r8, 6);
                let o19 = vgetq_lane_u8(g8, 6);
                let o20 = vgetq_lane_u8(b8, 6);
                let o21 = vgetq_lane_u8(r8, 7);
                let o22 = vgetq_lane_u8(g8, 7);
                let o23 = vgetq_lane_u8(b8, 7);
                *row_out.add(di + 18) = o18;
                *row_out.add(di + 19) = o19;
                *row_out.add(di + 20) = o20;
                *row_out.add(di + 21) = o21;
                *row_out.add(di + 22) = o22;
                *row_out.add(di + 23) = o23;
            }

            let col_start = pixel_groups * 8;
            let mut uv_idx = col_start / 2;
            let mut col = col_start;
            while col < width {
                let u = *u_row.add(uv_idx) as i32 - 128;
                let v = *v_row.add(uv_idx) as i32 - 128;
                uv_idx += 1;
                let uv_mul_v = K_V_R * v;
                let uv_mul_gv = K_V_G * v;
                let uv_mul_gu = K_U_G * u;
                let uv_mul_b = K_U_B * u;
                for px in 0..2 {
                    if col + px >= width {
                        break;
                    }
                    let y = *y_row.add(col + px) as i32;
                    let dx = if mirror {
                        width - 1 - (col + px)
                    } else {
                        col + px
                    };
                    let o = (row * width + dx) * 3;
                    let base = K_Y * (y - 16);
                    *row_out.add(o) = clamp8(div_scale(base + uv_mul_v));
                    *row_out.add(o + 1) = clamp8(div_scale(base - uv_mul_gv - uv_mul_gu));
                    *row_out.add(o + 2) = clamp8(div_scale(base + uv_mul_b));
                }
                col += 2;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn fast_path_available() -> bool {
    std::env::var_os("CAMERAD_FORCE_SCALAR").is_none()
        && std::arch::is_aarch64_feature_detected!("neon")
}

fn yuv420_to_rgb8_scalar(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    mirror: bool,
) {
    for row in 0..height {
        let y_row = &y_plane[row * width..][..width];
        let row_base = row * width;
        let mut uv_idx = 0usize;
        let mut col = 0usize;
        while col < width {
            let u = u_plane[uv_idx] as i32 - 128;
            let v = v_plane[uv_idx] as i32 - 128;
            uv_idx += 1;
            let uv_mul_v = K_V_R * v;
            let uv_mul_gv = K_V_G * v;
            let uv_mul_gu = K_U_G * u;
            let uv_mul_b = K_U_B * u;
            for px in 0..2 {
                if col + px >= width {
                    break;
                }
                let y = y_row[col + px] as i32;
                let dx = if mirror {
                    width - 1 - (col + px)
                } else {
                    col + px
                };
                let o = (row_base + dx) * 3;
                let base = K_Y * (y - 16);
                dst[o] = clamp8(div_scale(base + uv_mul_v));
                dst[o + 1] = clamp8(div_scale(base - uv_mul_gv - uv_mul_gu));
                dst[o + 2] = clamp8(div_scale(base + uv_mul_b));
            }
            col += 2;
        }
    }
}

pub fn yuv420_to_rgb8(src: &[u8], dst: &mut [u8], width: usize, height: usize, mirror: bool) {
    let y_size = width * height;
    let uv_size = (width / 2) * (height / 2);
    assert!(src.len() >= y_size + uv_size * 2);
    assert!(dst.len() >= width * height * 3);
    let y_plane = &src[..y_size];
    let u_plane = &src[y_size..][..uv_size];
    let v_plane = &src[y_size + uv_size..][..uv_size];

    #[cfg(target_arch = "aarch64")]
    if width >= 8 && width.is_multiple_of(8) && fast_path_available() {
        unsafe {
            neon::yuv420_to_rgb8_neon(
                y_plane.as_ptr(),
                u_plane.as_ptr(),
                v_plane.as_ptr(),
                dst.as_mut_ptr(),
                width,
                height,
                mirror,
            );
        }
        return;
    }
    yuv420_to_rgb8_scalar(y_plane, u_plane, v_plane, dst, width, height, mirror);
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
    if width >= 4 && width.is_multiple_of(4) && fast_path_available() {
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
    fn yuv420_red_pixel() {
        let w = 2usize;
        let h = 2usize;
        let uv = (w / 2) * (h / 2);
        let mut frame = vec![0u8; w * h + uv * 2];
        for y in frame.iter_mut().take(w * h) {
            *y = 80;
        }
        // U=0 (neutral, no blue) at y_size
        frame[w * h] = 0;
        // V=255 (max red) at y_size + uv
        frame[w * h + uv] = 255;
        let mut dst = vec![0u8; w * h * 3];
        super::yuv420_to_rgb8(&frame, &mut dst, w, h, false);
        for px in 0..4 {
            let r = dst[px * 3];
            let g = dst[px * 3 + 1];
            let b = dst[px * 3 + 2];
            assert!(r > g, "R={} should be > G={}", r, g);
            assert!(r > b, "R={} should be > B={}", r, b);
        }
    }

    #[test]
    fn yuv420_neutral_is_gray() {
        let w = 4usize;
        let h = 2usize;
        let uv = (w / 2) * (h / 2);
        let frame = vec![128u8; w * h + uv * 2];
        let mut dst = vec![0u8; w * h * 3];
        super::yuv420_to_rgb8(&frame, &mut dst, w, h, false);
        for px in 0..w * h {
            let r = dst[px * 3];
            let g = dst[px * 3 + 1];
            let b = dst[px * 3 + 2];
            assert_eq!(r, g, "gray: R={} != G={}", r, g);
            assert_eq!(g, b, "gray: G={} != B={}", g, b);
        }
    }

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
    fn yuv420_neon_matches_scalar_byte_exact() {
        let mut rng = 0xdead_beef_cafe_babeu64;
        let next = |rng: &mut u64| {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        };
        let mut mismatches = 0usize;
        for trial in 0..64 {
            for w in [8usize, 16, 32] {
                let h = 8usize;
                let y_size = w * h;
                let uv_size = (w / 2) * (h / 2);
                let mut src = vec![0u8; y_size + uv_size * 2];
                for b in src.iter_mut() {
                    next(&mut rng);
                    *b = if trial < 16 {
                        (rng >> 56) as u8
                    } else {
                        (rng % 179) as u8
                    };
                }
                for mirror in [false, true] {
                    let mut fast = vec![0u8; w * h * 3];
                    let mut scalar = vec![0u8; w * h * 3];
                    super::yuv420_to_rgb8(&src, &mut fast, w, h, mirror);
                    yuv420_to_rgb8_scalar(
                        &src[..y_size],
                        &src[y_size..][..uv_size],
                        &src[y_size + uv_size..][..uv_size],
                        &mut scalar,
                        w,
                        h,
                        mirror,
                    );
                    for i in 0..fast.len() {
                        if fast[i] != scalar[i] {
                            mismatches += 1;
                            if mismatches <= 5 {
                                let px = i / 3;
                                let ch = i % 3;
                                panic!(
                                    "yuv420 mismatch trial={trial} mirror={mirror} w={w} px={px} ch={ch}: neon={} scalar={}",
                                    fast[i], scalar[i]
                                );
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "yuv420 neon/scalar diverged on {mismatches} bytes"
        );
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
            let mut src = [0u8; 16 * 2 * 4];
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
