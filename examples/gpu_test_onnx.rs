use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig};
use vecboost::domain::BatchEmbedRequest;
use vecboost::engine::AnyEngine;
use vecboost::service::embedding::EmbeddingService;

const TEST_TEXTS: &[&str] = &[
    "人工智能正在改变我们的生活方式",
    "机器学习是实现人工智能的关键技术",
    "深度学习在计算机视觉领域取得了巨大突破",
    "自然语言处理让计算机能够理解人类语言",
    "向量数据库是AI应用的重要组成部分",
];

const LOCAL_MODEL_PATH: &str = "/home/dev/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181";

#[tokio::main]
async fn main() {
    println!("=== VecBoost GPU 向量化测试 (ONNX引擎) ===\n");

    println!("1. 读取配置文件...");
    let config = vecboost::AppConfig::load().expect("Failed to load config");
    println!("   use_gpu = {}", config.model.use_gpu);

    println!("\n2. 创建模型配置 (使用ONNX引擎)...");
    let model_path = PathBuf::from(LOCAL_MODEL_PATH);
    let model_config = ModelConfig {
        name: config.model.model_repo.clone(),
        engine_type: EngineType::Onnx,
        model_path: model_path.clone(),
        tokenizer_path: None,
        device: if config.model.use_gpu {
            DeviceType::Cuda
        } else {
            DeviceType::Cpu
        },
        max_batch_size: config.model.batch_size,
        pooling_mode: None,
        expected_dimension: config.model.expected_dimension,
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
    };
    println!("   设备类型: {:?}", model_config.device);
    println!("   引擎类型: {:?}", model_config.engine_type);
    println!("   模型路径: {:?}", model_path);

    println!("\n3. 初始化推理引擎 (这可能需要一些时间来下载模型)...");
    let start = Instant::now();
    let engine = AnyEngine::new(
        &model_config,
        EngineType::Onnx,
        vecboost::config::model::Precision::Fp32,
    )
    .expect("Failed to create engine");
    let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(engine));
    let init_time = start.elapsed();
    println!("   引擎初始化耗时: {:?}", init_time);

    println!("\n4. 创建Embedding服务...");
    let service = EmbeddingService::new(engine, Some(model_config));
    let service = Arc::new(RwLock::new(service));

    println!("\n5. 执行向量化测试...");
    println!("   测试文本数量: {}", TEST_TEXTS.len());
    let start = Instant::now();
    let service_guard = service.read().await;
    let texts_vec: Vec<String> = TEST_TEXTS.iter().map(|s| s.to_string()).collect();
    let result = service_guard
        .process_batch(BatchEmbedRequest {
            texts: texts_vec,
            mode: None,
        })
        .await;
    let embed_time = start.elapsed();
    drop(service_guard);

    match result {
        Ok(batch_response) => {
            println!("   向量化耗时: {:?}", embed_time);
            println!("   成功处理的文本数量: {}", batch_response.embeddings.len());
            println!("   向量维度: {}", batch_response.dimension);
            if let Some(first_embedding) = batch_response.embeddings.first() {
                println!("   第一个向量长度: {}", first_embedding.embedding.len());
            }
        }
        Err(e) => {
            println!("   向量化失败: {:?}", e);
        }
    }

    println!("\n=== 测试完成 ===");
    println!("\n当前状态:");
    println!("  - 配置中的 use_gpu: {}", config.model.use_gpu);
    println!(
        "  - 实际使用的设备: {}",
        if config.model.use_gpu {
            "GPU (CUDA)"
        } else {
            "CPU"
        }
    );

    if !config.model.use_gpu {
        println!("\n⚠️  当前未使用GPU加速！");
        println!("\n要启用GPU加速，请修改 config.toml:");
        println!("   [model]");
        println!("   use_gpu = true");
        println!("\n然后重新编译并运行:");
        println!("   cargo build");
        println!("   cargo run --example gpu_test_onnx");
    } else {
        println!("\n✅ GPU加速已启用！");
        println!("\n可以使用 nvidia-smi 命令监控GPU使用情况:");
        println!("   watch -n 1 nvidia-smi");
    }
}
