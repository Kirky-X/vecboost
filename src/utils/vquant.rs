// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 向量输出量化（port 自 colibri `kv_tq.h` 旋转技巧）。
//!
//! - Hadamard 旋转：splitmix32 符号向量 + 迭代 FWHT，无矩阵存储 O(n log n)；
//!   形式 `S·H·S/√n`（两侧符号）保证自逆性与范数保持。
//! - `quantize_i8`：旋转后 per-vector amax 对称量化；`dot_i8` 还原点积。
//! - `quantize_binary`：中位数校准符号位；`cosine_binary` 经 Hamming 估计余弦。
//! - 对外 embed API 输出保持 f32；量化仅限内部存储/比较路径（SemanticCache 粗筛）。

/// 支持旋转的块维度。
const SUPPORTED_DIMS: [usize; 3] = [256, 128, 64];

fn splitmix32(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 生成确定性符号向量（+1.0/-1.0），种子与维度/块位置绑定，编解码一致。
fn hadamard_signs(dim: usize, block_idx: u64) -> Vec<f32> {
    let mut state = 0x1234_5678_9ABC_DEF0u64
        ^ ((dim as u64) << 32)
        ^ block_idx.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..dim)
        .map(|_| {
            if splitmix32(&mut state) & 1 == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

fn fwht_inplace(a: &mut [f32]) {
    let mut h = 1usize;
    while h < a.len() {
        for i in (0..a.len()).step_by(2 * h) {
            for j in 0..h {
                let u = a[i + j];
                let v = a[i + j + h];
                a[i + j] = u + v;
                a[i + j + h] = u - v;
            }
        }
        h *= 2;
    }
}

/// 单块 Hadamard 旋转 `S·H·S/√n`（自逆、正交）。
pub fn hadamard_block(x: &[f32], block_idx: u64) -> Vec<f32> {
    let n = x.len();
    assert!(
        SUPPORTED_DIMS.contains(&n),
        "hadamard_block 仅支持 {:?}，实际 {}",
        SUPPORTED_DIMS,
        n
    );
    let signs = hadamard_signs(n, block_idx);
    let mut t: Vec<f32> = x.iter().zip(signs.iter()).map(|(a, s)| a * s).collect();
    fwht_inplace(&mut t);
    let inv = 1.0 / (n as f32).sqrt();
    for (v, s) in t.iter_mut().zip(signs.iter()) {
        *v = *v * *s * inv;
    }
    t
}

/// 全向量旋转：按 256/128/64 贪心分块，尾部不足 64 维保持原样。
pub fn rotate(v: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len());
    let mut offset = 0usize;
    let mut block_idx = 0u64;
    while v.len() - offset >= 64 {
        let rest = v.len() - offset;
        let dim = if rest >= 256 {
            256
        } else if rest >= 128 {
            128
        } else {
            64
        };
        out.extend(hadamard_block(&v[offset..offset + dim], block_idx));
        offset += dim;
        block_idx += 1;
    }
    out.extend_from_slice(&v[offset..]);
    out
}

/// i8 对称量化向量（旋转后编码）。
#[derive(Debug, Clone)]
pub struct I8Vector {
    pub data: Vec<i8>,
    pub scale: f32,
}

/// 将 f32 向量旋转后做 per-vector amax 对称量化。
pub fn quantize_i8(v: &[f32]) -> I8Vector {
    let r = rotate(v);
    let amax = r.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if amax == 0.0 || !amax.is_finite() {
        return I8Vector {
            data: vec![0; v.len()],
            scale: 1.0,
        };
    }
    let scale = amax / 127.0;
    let data = r
        .iter()
        .map(|x| ((x / scale).round() as i32).clamp(-127, 127) as i8)
        .collect();
    I8Vector { data, scale }
}

/// i8 还原点积：`scale_a·scale_b·Σq`。
pub fn dot_i8(a: &I8Vector, b: &I8Vector) -> f32 {
    assert_eq!(a.data.len(), b.data.len(), "dot_i8 维度必须一致");
    let acc: i64 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| *x as i64 * *y as i64)
        .sum();
    acc as f32 * a.scale * b.scale
}

/// binary 量化向量（中位数校准符号位）。
#[derive(Debug, Clone)]
pub struct BinaryVector {
    pub words: Vec<u64>,
    pub dim: usize,
    pub median: f32,
}

fn median_of(mut xs: Vec<f32>) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = xs.len() / 2;
    if xs.len().is_multiple_of(2) {
        (xs[mid - 1] + xs[mid]) / 2.0
    } else {
        xs[mid]
    }
}

/// 旋转后减中位数取符号位。
pub fn quantize_binary(v: &[f32]) -> BinaryVector {
    let r = rotate(v);
    let median = median_of(r.clone());
    let mut words = vec![0u64; v.len().div_ceil(64)];
    for (i, x) in r.iter().enumerate() {
        if *x >= median {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    BinaryVector {
        words,
        dim: v.len(),
        median,
    }
}

/// Hamming 余弦估计：`cos(π·h/n)`。
pub fn cosine_binary(a: &BinaryVector, b: &BinaryVector) -> f32 {
    assert_eq!(a.dim, b.dim, "cosine_binary 维度必须一致");
    if a.dim == 0 {
        return 0.0;
    }
    let ham: u32 = a
        .words
        .iter()
        .zip(b.words.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    let frac = ham as f32 / a.dim as f32;
    (std::f32::consts::PI * frac).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(xs: &[f32]) -> f32 {
        xs.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn test_hadamard_self_inverse_and_norm() {
        for &n in &[64usize, 128, 256] {
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 + 11) % 97) as f32 / 97.0 - 0.5)
                .collect();
            let y = hadamard_block(&x, 0);
            let z = hadamard_block(&y, 0);
            let max_err = x
                .iter()
                .zip(z.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 1e-5, "n={} 自逆误差 {} 超出 1e-5", n, max_err);
            let rel = ((l2(&y) - l2(&x)) / l2(&x)).abs();
            assert!(rel < 1e-4, "n={} 范数相对变化 {} 超出 1e-4", n, rel);
        }
    }

    /// Box-Muller 高斯采样（确定性种子）。
    struct Gauss {
        state: u64,
        spare: Option<f64>,
    }

    impl Gauss {
        fn new(seed: u64) -> Self {
            Self {
                state: seed,
                spare: None,
            }
        }

        fn u53(&mut self) -> f64 {
            // 简易 LCG 取均匀分布
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn next(&mut self) -> f64 {
            if let Some(v) = self.spare.take() {
                return v;
            }
            let u1 = self.u53().max(1e-12);
            let u2 = self.u53();
            let r = (-2.0 * u1.ln()).sqrt();
            let t = 2.0 * std::f64::consts::PI * u2;
            self.spare = Some(r * t.sin());
            r * t.cos()
        }
    }

    fn rand_vec(g: &mut Gauss, n: usize) -> Vec<f32> {
        (0..n).map(|_| g.next() as f32).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let d = dot(a, b) as f64;
        let na = dot(a, a) as f64;
        let nb = dot(b, b) as f64;
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            d / (na.sqrt() * nb.sqrt())
        }
    }

    #[test]
    fn test_i8_dot_relative_error_within_2pct() {
        let mut g = Gauss::new(0xC0FFEE);
        let mut max_rel = 0.0f64;
        for _ in 0..32 {
            let a = rand_vec(&mut g, 384);
            let b = rand_vec(&mut g, 384);
            let exact = dot(&a, &b) as f64;
            let qa = quantize_i8(&a);
            let qb = quantize_i8(&b);
            let est = dot_i8(&qa, &qb) as f64;
            // 384 维高斯点积可能接近 0，相对误差改用“相对范数积”归一
            let denom = (dot(&a, &a) as f64).sqrt() * (dot(&b, &b) as f64).sqrt();
            let rel = ((est - exact) / denom).abs();
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel <= 0.02, "i8 还原点积相对误差 {} 超出 2%", max_rel);
    }

    fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
        fn ranks(v: &[f64]) -> Vec<f64> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
            let mut r = vec![0.0; v.len()];
            for (rank, &i) in idx.iter().enumerate() {
                r[i] = rank as f64;
            }
            r
        }
        let rx = ranks(xs);
        let ry = ranks(ys);
        let n = xs.len() as f64;
        let mx = rx.iter().sum::<f64>() / n;
        let my = ry.iter().sum::<f64>() / n;
        let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
        for i in 0..xs.len() {
            cov += (rx[i] - mx) * (ry[i] - my);
            vx += (rx[i] - mx).powi(2);
            vy += (ry[i] - my).powi(2);
        }
        if vx == 0.0 || vy == 0.0 {
            0.0
        } else {
            cov / (vx.sqrt() * vy.sqrt())
        }
    }

    #[test]
    fn test_binary_cosine_spearman_above_09() {
        // 构造余弦 spread 的语料：base 向量 + 多档噪声拷贝（近重复→高余弦，
        // 强噪声→近正交），外加一组精确重复（粗筛必须排首位）。
        // 评估限定在工作区 |true| ≥ 0.2：1-bit 估计器在 384 维下无法分辨
        // 近正交对的噪声级差异（±0.05 真值 vs ±0.08 估计噪声），而粗筛的
        // 运行区间正是高相似区；把正交噪声纳入排序会度量噪声而非估计器。
        let mut g = Gauss::new(0xB17A7);
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for _ in 0..6 {
            let base = rand_vec(&mut g, 384);
            vecs.push(base.clone());
            vecs.push(base.clone()); // 精确重复
            for &noise in &[0.01f64, 0.05, 0.1, 0.3, 1.0] {
                let noisy: Vec<f32> = base
                    .iter()
                    .map(|x| x + (g.next() as f32) * (noise as f32))
                    .collect();
                vecs.push(noisy);
            }
        }
        let codes: Vec<BinaryVector> = vecs.iter().map(|v| quantize_binary(v)).collect();
        // 精确重复必须估计为 1.0（无假阴性，粗筛不漏）。
        assert!(
            (cosine_binary(&codes[0], &codes[1]) - 1.0).abs() < 1e-6,
            "精确重复的 binary 估计必须为 1.0"
        );
        let mut exact = Vec::new();
        let mut est = Vec::new();
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let t = cosine(&vecs[i], &vecs[j]);
                if t >= 0.2 {
                    exact.push(t);
                    est.push(cosine_binary(&codes[i], &codes[j]) as f64);
                }
            }
        }
        assert!(
            exact.len() >= 50,
            "工作区样本过少（{}），测试设计有误",
            exact.len()
        );
        let rho = spearman(&exact, &est);
        assert!(rho >= 0.9, "binary 余弦估计 Spearman {} < 0.9", rho);
    }

    #[test]
    fn test_rotated_not_worse_than_unrotated() {
        // 各向异性数据：单个大离群值，未旋转时 amax 量化误差大。
        let mut g = Gauss::new(12345);
        let mut mse_rot = 0.0f64;
        let mut mse_raw = 0.0f64;
        let trials = 16;
        for _ in 0..trials {
            let mut a = rand_vec(&mut g, 256);
            let b = rand_vec(&mut g, 256);
            a[0] = 50.0;
            let exact = dot(&a, &b) as f64;
            // 旋转路径
            let est_rot = dot_i8(&quantize_i8(&a), &quantize_i8(&b)) as f64;
            // 未旋转对照（直接 amax 量化）
            let q = |v: &[f32]| -> (Vec<i8>, f32) {
                let amax = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-12);
                let s = amax / 127.0;
                (
                    v.iter()
                        .map(|x| ((x / s).round() as i32).clamp(-127, 127) as i8)
                        .collect(),
                    s,
                )
            };
            let (qa, sa) = q(&a);
            let (qb, sb) = q(&b);
            let acc: i64 = qa
                .iter()
                .zip(qb.iter())
                .map(|(x, y)| *x as i64 * *y as i64)
                .sum();
            let est_raw = acc as f64 * sa as f64 * sb as f64;
            mse_rot += (est_rot - exact).powi(2);
            mse_raw += (est_raw - exact).powi(2);
        }
        assert!(
            mse_rot <= mse_raw,
            "旋转后量化误差 {} 应不劣于未旋转 {}",
            mse_rot,
            mse_raw
        );
    }
}
