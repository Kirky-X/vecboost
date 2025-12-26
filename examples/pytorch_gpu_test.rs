use serde_json;
use tokenizers::Tokenizer as HfTokenizer;

const LOCAL_MODEL_PATH: &str = "/home/dev/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181";
const TEST_TEXT: &str = "hello world";

fn main() {
    println!("=== VecBoost GPU 测试 (PyTorch) ===\n");

    println!("1. 检查CUDA可用性...");
    let cuda_available = candle_core::utils::cuda_is_available();
    println!("   CUDA可用: {}", cuda_available);

    if !cuda_available {
        println!("\n⚠️ CUDA不可用，测试终止");
        return;
    }

    println!("\n2. 创建CUDA设备...");
    let device = match candle_core::Device::new_cuda(0) {
        Ok(d) => {
            println!("   CUDA设备创建成功");
            d
        }
        Err(e) => {
            println!("   CUDA设备创建失败: {}", e);
            return;
        }
    };

    println!("\n3. 加载模型配置...");
    let model_path = std::path::PathBuf::from(LOCAL_MODEL_PATH);
    let config_path = model_path.join("config.json");
    let config_content = std::fs::read_to_string(&config_path).expect("无法读取config.json");
    let config: serde_json::Value =
        serde_json::from_str(&config_content).expect("无法解析config.json");
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;
    let num_attention_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    println!("   词汇表大小: {}", vocab_size);
    println!("   隐藏层大小: {}", hidden_size);
    println!("   注意力头数: {}", num_attention_heads);
    println!("   层数: {}", num_hidden_layers);

    println!("\n4. 加载tokenizer...");
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = HfTokenizer::from_file(tokenizer_path.to_string_lossy().as_ref())
        .expect("无法加载tokenizer");
    println!("   Tokenizer加载成功");

    println!("\n5. 加载PyTorch模型权重...");
    let weights_path = model_path.join("pytorch_model.bin");
    let dtype = candle_core::DType::F32;

    let vb = candle_nn::VarBuilder::from_pth(&weights_path, dtype, &device)
        .expect("无法加载PyTorch权重");
    println!("   PyTorch权重加载成功");

    println!("\n6. 创建XLMRoberta模型...");
    let use_cache = config["use_cache"].as_bool().unwrap_or(false);
    let model_type = config["model_type"].as_str().unwrap_or("xlm-roberta");

    let do_use_cache = if model_type == "xlm-roberta" {
        use_cache
    } else {
        false
    };

    let attention_probs_dropout_prob = config["attention_probs_dropout_prob"]
        .as_f64()
        .unwrap_or(0.1) as f32;
    let attention_dropout_prob = attention_probs_dropout_prob;
    let embed_dropout_prob = config["hidden_dropout_prob"].as_f64().unwrap_or(0.1) as f32;
    let layer_norm_eps = config["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
    let max_position_embeddings =
        config["max_position_embeddings"].as_u64().unwrap_or(8194) as usize;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(4096) as usize;

    println!("\n7. 编码测试文本...");
    let encoding = tokenizer.encode(TEST_TEXT, true).expect("无法编码文本");
    let ids: Vec<u32> = encoding.get_ids().iter().map(|&x| x as u32).collect();
    let attention_mask: Vec<u32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| x as u32)
        .collect();
    println!("   原始Token数量: {}", ids.len());
    println!("   原始Tokens: {:?}", encoding.get_tokens());

    let max_id = ids.iter().max().copied().unwrap_or(0);
    let min_id = ids.iter().min().copied().unwrap_or(0);
    println!("   Token ID范围: [{}, {}]", min_id, max_id);

    if max_id >= vocab_size as u32 {
        println!("   ⚠️ 发现超出词汇表的token ID，进行过滤...");
        let valid_ids: Vec<u32> = ids
            .iter()
            .filter(|&&id| id < vocab_size as u32)
            .cloned()
            .collect();
        let valid_mask: Vec<u32> = attention_mask
            .iter()
            .filter(|&&id| id < vocab_size as u32 || id == 0)
            .cloned()
            .collect();
        println!("   过滤后Token数量: {}", valid_ids.len());

        if valid_ids.is_empty() {
            println!("   ⚠️ 没有有效token，测试终止");
            return;
        }
    }

    println!("\n8. 创建输入张量 (同步模式)...");
    let token_ids = candle_core::Tensor::new(ids.as_slice(), &device)
        .expect("无法创建token_ids张量")
        .unsqueeze(0)
        .expect("无法添加batch维度");
    let attention_mask_tensor = candle_core::Tensor::new(attention_mask.as_slice(), &device)
        .expect("无法创建attention_mask张量")
        .unsqueeze(0)
        .expect("无法添加batch维度");
    println!("   token_ids shape: {:?}", token_ids.shape());
    println!(
        "   attention_mask shape: {:?}",
        attention_mask_tensor.shape()
    );

    println!("\n9. 由于candle-transformers没有XLMRoberta支持，使用ONNX Runtime进行GPU测试...");

    println!("\n10. 检查ONNX Runtime GPU可用性...");
    let ort = ort::init()
        .providers(vec![ort::Provider::CUDA])
        .commit()
        .expect("无法初始化ONNX Runtime");

    let providers = ort::SessionBuilder::new()
        .expect("无法创建SessionBuilder")
        .with_optimization_level(ort::GraphOptimizationLevel::All)
        .expect("无法设置优化级别");

    println!("   ONNX Runtime初始化成功 (使用CUDA)");

    println!("\n=== 测试需要使用ONNX Runtime引擎 ===");
    println!("请使用主程序: cargo run --example gpu_test --features cuda --release");
}
