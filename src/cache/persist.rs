// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! embedding 缓存崩溃安全 WAL（port 自 colibri `kv_persist.h`）。
//!
//! 两段追加协议：先写数据记录（key + 向量 + 模型指纹 + checksum），后写提交
//! 记录（nrec+1）。崩溃时最多丢最后一条，绝不把半条当完整记录。
//!
//! 持久化保证的诚实边界（审查修正）：实现为 write 到 page cache +
//! 用户态缓冲 flush，**无 fsync**——仅防进程崩溃，不防断电/系统崩溃
//! （断电时丢失量可能超过一条，由 checksum 兜底不产生脏数据）；
//! checksum 为无密钥 xxh3_64 混合，不防本地篡改（信任边界为本地文件）。
//!
//! 回放语义：顺序回放，版本头/指纹不匹配或 checksum 失败 → 弃该记录并 warn
//! （中段损坏跳过继续；长度前缀损坏则视为撕裂尾，停止回放）；
//! 超过 `persist_max_bytes` 触发快照重写紧凑化（临时文件 + rename 原子替换）。
//! 不引入后台线程：追加在插入调用点同步完成。

use log::warn;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// 文件魔数 "VBC1"。
const FILE_MAGIC: [u8; 4] = *b"VBC1";
/// 文件格式版本。
const FILE_VERSION: u32 = 1;
/// 记录类型：数据。
const REC_DATA: u8 = 1;
/// 记录类型：提交。
const REC_COMMIT: u8 = 2;
/// 长度 sanity 上限（key/向量字节数超过即视为撕裂）。
const MAX_SANE_LEN: u32 = 64 * 1024 * 1024;

fn checksum(key: &[u8], vec_bytes: &[u8], tag: &[u8]) -> u64 {
    let mut h = xxhash_rust::xxh3::xxh3_64(key);
    h ^= xxhash_rust::xxh3::xxh3_64(vec_bytes).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= xxhash_rust::xxh3::xxh3_64(tag).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h
}

fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// 写文件头（新文件构造时调用一次；追加路径不再逐次 stat，见 审查）。
pub fn write_header(file: &mut impl std::io::Write) -> std::io::Result<()> {
    file.write_all(&FILE_MAGIC)?;
    write_u32(file, FILE_VERSION)?;
    file.flush()?;
    Ok(())
}

/// 追加一条数据记录 + 提交记录（调用方持有文件锁）。
/// 记录先序列化进调用方缓冲，合并为 **2 次 write**（审查
/// 逐字段写会产生 ~10 次系统调用/插入）。返回提交后的 nrec。
pub fn append_record<W: std::io::Write>(
    file: &mut W,
    key: &str,
    vec: &[f32],
    tag: &str,
    nrec: u64,
) -> std::io::Result<(u64, usize)> {
    // 借 f32 位模式写原始字节（逐位精确）。
    let key_b = key.as_bytes();
    let tag_b = tag.as_bytes();
    let mut vec_b = Vec::with_capacity(vec.len() * 4);
    for x in vec {
        vec_b.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    let sum = checksum(key_b, &vec_b, tag_b);

    let mut rec = Vec::with_capacity(1 + 4 + key_b.len() + 4 + vec_b.len() + 4 + tag_b.len() + 8);
    rec.push(REC_DATA);
    rec.extend_from_slice(&(key_b.len() as u32).to_le_bytes());
    rec.extend_from_slice(key_b);
    rec.extend_from_slice(&(vec.len() as u32).to_le_bytes());
    rec.extend_from_slice(&vec_b);
    rec.extend_from_slice(&(tag_b.len() as u32).to_le_bytes());
    rec.extend_from_slice(tag_b);
    rec.extend_from_slice(&sum.to_le_bytes());
    file.write_all(&rec)?;

    let nrec = nrec + 1;
    let mut commit = Vec::with_capacity(9);
    commit.push(REC_COMMIT);
    commit.extend_from_slice(&nrec.to_le_bytes());
    file.write_all(&commit)?;
    Ok((nrec, rec.len() + commit.len()))
}

/// 回放文件：返回 `(有效提交记录, 最后提交计数)`。
/// 仅返回通过 checksum 且指纹（tag）匹配的记录（顺序，后者覆盖前者由调用方处理）。
pub fn replay(path: &Path, expected_tag: &str) -> (Vec<(String, Vec<f32>)>, u64) {
    let mut out = Vec::new();
    let mut committed: u64 = 0;
    let mut valid: Vec<(String, Vec<f32>)> = Vec::new();
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            warn!("persist: 无法打开 {}（{}），视为空缓存", path.display(), e);
            return (out, committed);
        }
    };
    // 文件头。
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return (out, committed); // 空文件
    }
    if magic != FILE_MAGIC {
        warn!("persist: {} 魔数不匹配，整体弃用", path.display());
        return (out, committed);
    }
    match read_u32(&mut file) {
        Ok(v) if v == FILE_VERSION => {}
        Ok(v) => {
            warn!("persist: {} 版本 {} 不匹配，整体弃用", path.display(), v);
            return (out, committed);
        }
        Err(_) => return (out, committed), // 头撕裂
    }
    loop {
        let mut t = [0u8; 1];
        match file.read_exact(&mut t) {
            Ok(()) => {}
            Err(_) => break, // 正常 EOF 或撕裂尾
        }
        if t[0] == REC_COMMIT {
            match read_u64(&mut file) {
                Ok(n) => committed = committed.max(n),
                Err(_) => break, // 提交记录半条 → 撕裂尾，丢弃
            }
            continue;
        }
        if t[0] != REC_DATA {
            warn!(
                "persist: {} 未知记录类型 {}，停止回放（撕裂）",
                path.display(),
                t[0]
            );
            break;
        }
        // 数据记录：长度先行，超限即视为撕裂尾。
        let key_len = match read_u32(&mut file) {
            Ok(n) if n <= MAX_SANE_LEN => n as usize,
            _ => break,
        };
        let mut key_b = vec![0u8; key_len];
        if file.read_exact(&mut key_b).is_err() {
            break;
        }
        let dim = match read_u32(&mut file) {
            Ok(n) if n <= MAX_SANE_LEN / 4 => n as usize,
            _ => break,
        };
        let mut vec_b = vec![0u8; dim * 4];
        if file.read_exact(&mut vec_b).is_err() {
            break;
        }
        let tag_len = match read_u32(&mut file) {
            Ok(n) if n <= MAX_SANE_LEN => n as usize,
            _ => break,
        };
        let mut tag_b = vec![0u8; tag_len];
        if file.read_exact(&mut tag_b).is_err() {
            break;
        }
        let mut sum_b = [0u8; 8];
        if file.read_exact(&mut sum_b).is_err() {
            break;
        }
        // 校验：checksum/指纹/UTF-8，失败则弃该条、继续。
        if u64::from_le_bytes(sum_b) != checksum(&key_b, &vec_b, &tag_b) {
            warn!("persist: {} 一条记录 checksum 失败，已跳过", path.display());
            continue;
        }
        if tag_b != expected_tag.as_bytes() {
            warn!("persist: {} 一条记录指纹不匹配，已跳过", path.display());
            continue;
        }
        let key = match String::from_utf8(key_b) {
            Ok(k) => k,
            Err(_) => {
                warn!("persist: {} 一条记录 key 非 UTF-8，已跳过", path.display());
                continue;
            }
        };
        let mut v = Vec::with_capacity(dim);
        for chunk in vec_b.chunks_exact(4) {
            v.push(f32::from_bits(u32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ])));
        }
        valid.push((key, v));
    }
    // 仅应用已提交前缀（崩溃撕裂尾 beyond committed 被丢弃）。
    out.extend(valid.into_iter().take(committed as usize));
    (out, committed)
}

/// 快照重写紧凑化（tmp + rename 原子替换）。`entries` 为去重后全集。
pub fn compact(
    path: &Path,
    entries: &HashMap<String, Vec<f32>>,
    tag: &str,
) -> std::io::Result<u64> {
    let tmp = path.with_extension("compact.tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&FILE_MAGIC)?;
        write_u32(&mut file, FILE_VERSION)?;
        let mut nrec = 0u64;
        let mut keys: Vec<&String> = entries.keys().collect();
        keys.sort();
        for key in keys {
            nrec = append_record(&mut file, key, &entries[key], tag, nrec)?.0;
        }
        file.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(entries.len() as u64)
}

/// 测试辅助：统计文件内数据记录数与最后提交计数（不校验内容）。
#[cfg(test)]
pub fn record_counts(path: &Path) -> (usize, u64) {
    let mut data = 0usize;
    let mut last_commit = 0u64;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return (0, 0),
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return (0, 0);
    }
    if read_u32(&mut file).is_err() {
        return (0, 0);
    }
    loop {
        let mut t = [0u8; 1];
        if file.read_exact(&mut t).is_err() {
            break;
        }
        if t[0] == REC_COMMIT {
            match read_u64(&mut file) {
                Ok(n) => last_commit = n,
                Err(_) => break,
            }
            continue;
        }
        if t[0] != REC_DATA {
            break;
        }
        // 版式：[key_len][key][dim][vec][tag_len][tag][sum] —— 必须按序跳过，
        // 不能在 key/vec 载荷位置误读长度前缀。
        use std::io::Seek;
        let key_len = match read_u32(&mut file) {
            Ok(n) => n as u64,
            Err(_) => break,
        };
        if file
            .seek(std::io::SeekFrom::Current(key_len as i64))
            .is_err()
        {
            break;
        }
        let dim = match read_u32(&mut file) {
            Ok(n) => n as u64,
            Err(_) => break,
        };
        if file
            .seek(std::io::SeekFrom::Current((dim * 4) as i64))
            .is_err()
        {
            break;
        }
        let tag_len = match read_u32(&mut file) {
            Ok(n) => n as u64,
            Err(_) => break,
        };
        if file
            .seek(std::io::SeekFrom::Current((tag_len + 8) as i64))
            .is_err()
        {
            break;
        }
        data += 1;
    }
    (data, last_commit)
}
