//! 嵌入 torch 后端（TRANSCRIBE_BACKEND=py-torch）：PyO3 内嵌 CPython，
//! 进程内调用 qwen_asr/transformers 在 GPU 上跑 1.7B bf16 原生精度。
//!
//! 嵌入前提（main.rs 在 Python 初始化前设置环境变量）：
//! - PYTHONHOME → uv 独立 Python 前缀（含 libpython 与标准库）
//! - PYTHONPATH → .venv-export 的 site-packages（torch/transformers/qwen_asr）
//!
//! glue 脚本见 tools/torch_backend_glue.py（include_str 编译期嵌入）。

use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// 本地推理实现已并入主仓（transcribe-poc-shell-a）；基线通路自足错误类型。
type AsrResult<T> = Result<T, Box<dyn std::error::Error>>;

const GLUE_SRC: &str = include_str!("../../tools/torch_backend_glue.py");

/// 整段转录（qwen_asr 内部自行处理长音频切块，不经本地 VAD/fbank 管线）。
/// 一次性调用：模型以 Python 模块级单例缓存，进程生命周期内只加载一次。
pub fn run_pytorch_baseline(wav: &Path, model_dir: &str) -> AsrResult<String> {
    Python::with_gil(|py| {
        let glue = PyModule::from_code_bound(
            py,
            GLUE_SRC,
            "torch_backend_glue.py",
            "torch_backend_glue",
        )?;
        glue.call_method1("ensure_model", (model_dir,))?;
        let text: String = glue
            .call_method1("transcribe", (wav.to_string_lossy().as_ref(), model_dir))?
            .extract()?;
        Ok(text)
    })
}
