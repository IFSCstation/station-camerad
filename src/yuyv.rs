const SCALE: i32 = 1024;
const K_Y: i32 = 1192;
const K_V_R: i32 = 1634;
const K_V_G: i32 = 833;
const K_U_G: i32 = 400;
const K_U_B: i32 = 2066;

fn clamp8(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

fn convert(y: i32, u: i32, v: i32) -> [u8; 3] {
    let base = K_Y * (y - 16);
    let r = (base + K_V_R * v) / SCALE;
    let g = (base - K_V_G * v - K_U_G * u) / SCALE;
    let b = (base + K_U_B * u) / SCALE;
    [clamp8(r), clamp8(g), clamp8(b)]
}

pub fn yuyv_to_rgb8(src: &[u8], dst: &mut [u8], width: usize, height: usize, mirror: bool) {
    assert_eq!(width % 2, 0);
    assert!(src.len() >= width * height * 2);
    assert!(dst.len() >= width * height * 3);

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

#[cfg(test)]
mod tests {
    use super::*;

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
        let src = [82, 200, 150, 90]; // U=200 high, V=90 low
        let mut dst = [0u8; 6];
        yuyv_to_rgb8(&src, &mut dst, 2, 1, false);
        assert!(dst[2] > dst[0], "expected blue-ish, got {dst:?}");
        assert!(dst[5] > dst[3], "expected blue-ish, got {dst:?}");
    }
}
