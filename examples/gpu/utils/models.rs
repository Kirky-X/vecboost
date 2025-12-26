use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};

pub const DEFAULT_MODEL_PATHS: &[&str] = &[
    "/home/project/vecboost/models/bge-m3-onnx",
    "/home/dev/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
    "./model",
];

pub fn create_onnx_config(
    model_path: String,
    use_gpu: bool,
    batch_size: usize,
    expected_dimension: usize,
) -> ModelConfig {
    ModelConfig {
        name: "bge-m3".to_string(),
        engine_type: EngineType::Onnx,
        model_path,
        tokenizer_path: None,
        device: if use_gpu {
            DeviceType::Cuda
        } else {
            DeviceType::Cpu
        },
        max_batch_size: batch_size,
        pooling_mode: None,
        expected_dimension,
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
    }
}

pub fn create_candle_config(
    model_path: String,
    use_gpu: bool,
    batch_size: usize,
    expected_dimension: usize,
    precision: Precision,
) -> ModelConfig {
    ModelConfig {
        name: "bge-m3".to_string(),
        engine_type: EngineType::Candle,
        model_path,
        tokenizer_path: None,
        device: if use_gpu {
            DeviceType::Cuda
        } else {
            DeviceType::Cpu
        },
        max_batch_size: batch_size,
        pooling_mode: None,
        expected_dimension,
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
    }
}

pub fn get_test_texts() -> Vec<String> {
    vec![
        "这是一个测试文本，用于验证向量嵌入模型的性能。".to_string(),
        "VecBoost是一个高性能的向量嵌入服务框架。".to_string(),
        "GPU加速可以显著提升推理速度。".to_string(),
    ]
}

pub fn get_single_test_text() -> String {
    "这是一个测试文本，用于验证向量嵌入模型的性能。".to_string()
}
