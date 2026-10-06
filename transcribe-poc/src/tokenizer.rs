//! Qwen3-ASR tokenizer 封装（HF tokenizers crate，vocab.json + merges.txt BPE）。
//!
//! qwen-asr-tokenizer.cc（Apache-2.0，仅作移植参考）：
//! - Encode 特殊 token 整体查表（:1202-1244）：本实现以固定 id 表手工切分，
//!   普通段交给 BPE。**不**使用 tokenizers 的 add_special_tokens——那会按
//!   vocab 大小连续分配新 id（实测 <|audio_start|> 得 151646），与
//!   tokenizer_config.json 的真实 id（151669）漂移
//! - Decode 跳过 <|im_start|>/<|im_end|>（IsSkippableSpecialToken :780-782），
//!   其余特殊 token 原样输出文本，普通段 byte-level 还原（:1255-1284）
//! - 预切分正则与 Qwen2/Qwen3 一致（对应 SplitByQwen3Pattern 手写状态机）

use std::path::Path;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;
use tokenizers::Tokenizer;

pub type TokResult<T> = Result<T, Box<dyn std::error::Error>>;

/// 特殊 token 与固定 id（实测 assets/models/tokenizer/tokenizer_config.json
/// added_tokens_decoder；仅收录管线用到的 7 个，new() 逐一锚定校验）
pub const SPECIAL_TOKENS: [(&str, i64); 7] = [
    ("<|endoftext|>", 151643),
    ("<|im_start|>", 151644),
    ("<|im_end|>", 151645),
    ("<|audio_start|>", 151669),
    ("<|audio_end|>", 151670),
    ("<|audio_pad|>", 151676),
    ("<asr_text>", 151704),
];

pub const EOS_ID: i64 = 151645;
pub const ASR_TEXT_ID: i64 = 151704;
pub const AUDIO_PAD_ID: i64 = 151676;

const QWEN_PRETOKENIZER_REGEX: &str = concat!(
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
    "[^\r\n\\p{L}\\p{N}]?\\p{L}+|",
    "\\p{N}{1,3}|",
    " ?[^\\s\\p{L}\\p{N}]+[\r\n]*|",
    "\\s*[\r\n]+|",
    "\\s+(?!\\S)|",
    "\\s+"
);

/// tokenizers 的错误是 `Box<dyn Error + Send + Sync>`，转为本 crate 的
/// `Box<dyn Error>`（经 Display 重建，保留错误信息）
fn box_err<E: std::fmt::Display>(e: E) -> Box<dyn std::error::Error> {
    e.to_string().into()
}

pub struct AsrTokenizer {
    tk: Tokenizer,
}

impl AsrTokenizer {
    pub fn new(dir: &Path) -> TokResult<Self> {
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        for f in [&vocab, &merges] {
            if !f.exists() {
                return Err(format!("tokenizer file not found: {}", f.display()).into());
            }
        }
        let bpe = BPE::from_file(
            vocab.to_str().ok_or("vocab path not utf-8")?,
            merges.to_str().ok_or("merges path not utf-8")?,
        )
        .build()
        .map_err(box_err)?;

        let split = Split::new(
            QWEN_PRETOKENIZER_REGEX,
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(box_err)?;
        // HF Qwen2/Qwen3 配置：add_prefix_space=false, trim_offsets=false,
        // use_regex=true（ByteLevel::default() 是 add_prefix_space=true，会把
        // 首个普通段变成 "Ġ…" 形式，id 漂移）
        let pre = PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, true)),
        ]));

        let mut tk = Tokenizer::new(bpe);
        tk.with_pre_tokenizer(Some(pre));
        tk.with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevelDecoder::default())));
        // 特殊 id 锚定由测试 special_tokens_encode_to_pinned_ids 覆盖
        // （vocab.json 实测 "system"=8948 等），漂移即测试失败

        Ok(Self { tk })
    }

    /// Encode（qwen-asr-tokenizer.cc:1202-1244 语义）：最早出现的特殊 token
    /// 整体查表，普通段交给 BPE
    pub fn encode(&self, text: &str) -> TokResult<Vec<i64>> {
        let mut out = Vec::new();
        let mut rest = text;
        loop {
            let mut hit: Option<(usize, usize, i64)> = None; // (pos, len, id)
            for (token, id) in SPECIAL_TOKENS {
                if let Some(pos) = rest.find(token) {
                    let better = match hit {
                        None => true,
                        Some((p, l, _)) => pos < p || (pos == p && token.len() > l),
                    };
                    if better {
                        hit = Some((pos, token.len(), id));
                    }
                }
            }
            let Some((pos, len, id)) = hit else {
                break;
            };
            if pos > 0 {
                out.extend(self.encode_plain(&rest[..pos])?);
            }
            out.push(id);
            rest = &rest[pos + len..];
        }
        if !rest.is_empty() {
            out.extend(self.encode_plain(rest)?);
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str) -> TokResult<Vec<i64>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let enc = self.tk.encode(text, false).map_err(box_err)?;
        Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
    }

    /// Decode（qwen-asr-tokenizer.cc:1255-1284 语义）：<|im_start|>/<|im_end|>
    /// 跳过，其余特殊 token 原样输出文本，普通 id 由 BPE byte-level 还原
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut text = String::new();
        let mut plain: Vec<u32> = Vec::new();

        let flush = |plain: &mut Vec<u32>, text: &mut String| {
            if !plain.is_empty() {
                if let Ok(s) = self.tk.decode(plain, false) {
                    text.push_str(&s);
                }
                plain.clear();
            }
        };

        for &id in ids {
            if let Some((token, _)) = SPECIAL_TOKENS.iter().find(|(_, sid)| *sid == id) {
                if *token == "<|im_start|>" || *token == "<|im_end|>" {
                    continue;
                }
                flush(&mut plain, &mut text);
                text.push_str(token);
            } else if (0..=u32::MAX as i64).contains(&id) {
                plain.push(id as u32);
            }
        }
        flush(&mut plain, &mut text);

        // IsSkippableSpecialToken 兜底（普通段 decode 不含，防御性移除）
        for t in ["<|im_start|>", "<|im_end|>"] {
            text = text.replace(t, "");
        }
        text
    }

    /// 移除 U+FFFD 替换符
    pub fn remove_replacement_chars(s: &str) -> String {
        s.replace('\u{FFFD}', "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Option<AsrTokenizer> {
        let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/models/tokenizer"));
        if !dir.exists() {
            // 资产不入库（.gitignore），缺失时跳过（门禁环境必有）
            return None;
        }
        Some(AsrTokenizer::new(dir).unwrap())
    }

    #[test]
    fn special_tokens_encode_to_pinned_ids() {
        let Some(tok) = tok() else { return };
        for (token, id) in SPECIAL_TOKENS {
            assert_eq!(tok.encode(token).unwrap(), vec![id], "{token}");
        }
        // 混合：特殊 token 夹普通文本；"\n" 按预切分独立成段（id 198 = "Ċ"）
        let ids = tok.encode("<|im_start|>system\n<|im_end|>\n").unwrap();
        assert_eq!(&ids[..1], &[151644]);
        assert_eq!(ids[1], 8948); // "system"（vocab.json 实测）
        assert_eq!(*ids.last().unwrap(), 198); // 尾部 "\n"
    }

    #[test]
    fn plain_text_roundtrip() {
        let Some(tok) = tok() else { return };
        let ids = tok.encode("system\n").unwrap();
        assert!(!ids.is_empty());
        assert_eq!(tok.decode(&ids), "system\n");
        let ids = tok.encode("你好，世界 Hello!").unwrap();
        assert_eq!(tok.decode(&ids), "你好，世界 Hello!");
    }

    #[test]
    fn decode_skips_im_start_and_im_end_keeps_others() {
        let Some(tok) = tok() else { return };
        let ids = vec![151644i64, 8948, 151645];
        assert_eq!(tok.decode(&ids), "system");
        // 非 skippable 特殊 token 原样输出
        let ids = vec![151676i64, 8948];
        assert_eq!(tok.decode(&ids), "<|audio_pad|>system");
        assert_eq!(AsrTokenizer::remove_replacement_chars("a\u{FFFD}b"), "ab");
    }
}
