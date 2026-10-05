"""嵌入 torch 后端的 Python glue：模型单例 + 整段转录（qwen_asr 内部处理长音频切块）。"""
import torch

_model = None


def ensure_model(model_dir: str):
    global _model
    if _model is None:
        from qwen_asr import Qwen3ASRModel
        _model = Qwen3ASRModel.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=8,
            max_new_tokens=4096,
        )
    return _model


def transcribe(wav_path: str, model_dir: str) -> str:
    m = ensure_model(model_dir)
    out = m.transcribe(audio=wav_path, language="Chinese")
    return out[0].text if out else ""
