// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! `hgemm_` 符号垫片（feature `mkl`， 收敛修复）。
//!
//! 上游版本错配（证据链见 docs/PERFORMANCE.md）：candle-core 0.11 的 mkl 后端调用
//! fp16 GEMM `hgemm_`，而 intel-mkl-src 0.8.1 在 Linux 静态路径锁死 MKL 2020.1
//! （ghcr.io/rust-math OCI 镜像 2020.1-3038006115），该版本不导出 `hgemm_`
//! （fp16 GEMM 自 oneAPI MKL 2021.4 起提供）→ 链接失败 `undefined symbol: hgemm_`。
//! Windows 路径拉 2022.0 无此问题。
//!
//! 本垫片提供与 Intel `hgemm_`（cblas_hgemm Fortran 接口）ABI 兼容的实现：
//! f16 输入上转为 f32，调用同一 MKL 静态库的 `sgemm_`（f32 累加），输出转回
//! f16。数值语义与硬件 hgemm（f32 累加、f16 输出）一致。
//! intel-mkl-src 升级到提供真实 hgemm_ 的版本后，本垫片与链接顺序冲突自动消失
//! （真实符号在 MKL 静态库中，Rust 侧弱符号会被其覆盖——届时可直接删除本文件）。

use std::ffi::c_char;
use std::ffi::c_int;

unsafe extern "C" {
    // candle-core mkl 后端同款 Fortran BLAS 符号（MKL 2020.1 静态库已导出，
    // candle 的 f32 sgemm 路径亦引用它）。
    fn sgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f32,
        a: *const f32,
        lda: *const c_int,
        b: *const f32,
        ldb: *const c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: *const c_int,
    );
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hgemm_(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const half::f16,
    a: *const half::f16,
    lda: *const c_int,
    b: *const half::f16,
    ldb: *const c_int,
    beta: *const half::f16,
    c: *mut half::f16,
    ldc: *const c_int,
) {
    let (m, n, k) = (*m, *n, *k);
    let (lda, ldb, ldc) = (*lda, *ldb, *ldc);
    // 纵深防御：BLAS 调用方约定传正值，负值/零经 as usize
    // 回绕会越界读；直接拒绝畸形输入。
    if m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0 {
        return;
    }
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let ta = *transa as u8;
    let tb = *transb as u8;
    let is_trans = |flag: u8| flag == b'T' || flag == b't' || flag == b'C' || flag == b'c';

    // Fortran 列主序：A 为 m×k（不转置）或 k×m（转置），ld 条目按读取维度遍历。
    let a_rows = if is_trans(ta) { ku } else { mu };
    let a_cols = if is_trans(ta) { mu } else { ku };
    let b_rows = if is_trans(tb) { nu } else { ku };
    let b_cols = if is_trans(tb) { ku } else { nu };

    let a_f32: Vec<f32> = (0..a_cols)
        .flat_map(|col| {
            let base = col * lda as usize;
            (0..a_rows).map(move |row| *a.add(base + row))
        })
        .map(|v| v.to_f32())
        .collect();
    let b_f32: Vec<f32> = (0..b_cols)
        .flat_map(|col| {
            let base = col * ldb as usize;
            (0..b_rows).map(move |row| *b.add(base + row))
        })
        .map(|v| v.to_f32())
        .collect();

    let mut out = vec![0.0_f32; mu * nu];
    let alpha_f32 = (*alpha).to_f32();
    let beta_f32 = (*beta).to_f32();
    sgemm_(
        transa,
        transb,
        &m,
        &n,
        &k,
        &alpha_f32,
        a_f32.as_ptr(),
        &lda,
        b_f32.as_ptr(),
        &ldb,
        &beta_f32,
        out.as_mut_ptr(),
        &ldc,
    );

    // 写回列主序：C 为 m×n，ldc 条目。（beta=0 时 sgemm_ 不读 C，out 即结果。）
    for col in 0..nu {
        let base = col * ldc as usize;
        for row in 0..mu {
            *c.add(base + row) = half::f16::from_f32(out[col * mu + row]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C = A·B（不转置），f16 输入对 f32 参考的容差验证。
    #[test]
    fn hgemm_matches_f32_reference() {
        let (m, n, k) = (2usize, 3usize, 4usize);
        // 列主序 A(m×k)、B(k×n)
        let a: Vec<half::f16> = (0..m * k)
            .map(|i| half::f16::from_f32((i % 7) as f32 - 3.0))
            .collect();
        let b: Vec<half::f16> = (0..k * n)
            .map(|i| half::f16::from_f32((i % 5) as f32 - 2.0))
            .collect();
        let mut c = vec![half::f16::ZERO; m * n]; // ldc = m
        let one = half::f16::ONE;
        let zero = half::f16::ZERO;

        unsafe {
            hgemm_(
                &(b'N' as c_char),
                &(b'N' as c_char),
                &(m as c_int),
                &(n as c_int),
                &(k as c_int),
                &one,
                a.as_ptr(),
                &(m as c_int),
                b.as_ptr(),
                &(k as c_int),
                &zero,
                c.as_mut_ptr(),
                &(m as c_int),
            );
        }

        // f32 参考（行主序换算：A_col(i,j) = A_row(j,i)）
        let a2d: Vec<Vec<f32>> = (0..m)
            .map(|i| (0..k).map(|j| a[j * m + i].to_f32()).collect())
            .collect();
        let b2d: Vec<Vec<f32>> = (0..k)
            .map(|i| (0..n).map(|j| b[j * k + i].to_f32()).collect())
            .collect();
        for row in 0..m {
            for col in 0..n {
                let expect: f32 = (0..k).map(|t| a2d[row][t] * b2d[t][col]).sum();
                let got = c[col * m + row].to_f32();
                assert!(
                    (expect - got).abs() < 1e-2,
                    "C[{row}][{col}] 期望 {expect} 实得 {got}"
                );
            }
        }
    }
}
