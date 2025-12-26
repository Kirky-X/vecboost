use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use serde_json;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer as HfTokenizer;

const LOCAL_MODEL_PATH: &str = "/home/dev/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181";
const TEST_TEXT: &str = "hello world";

fn main() {
    println!("=== VecBoost 简单 GPU 测试 ===\n");

    println!("1. 检查CUDA可用性...");
    let cuda_available = candle_core::utils::cuda_is_available();
    println!("   CUDA可用: {}", cuda_available);

    if !cuda_available {
        println!("\n⚠️ CUDA不可用，测试终止");
        return;
    }

    println!("\n2. 创建CUDA设备...");
    let device = match Device::new_cuda(0) {
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
    let model_path = PathBuf::from(LOCAL_MODEL_PATH);
    let config_path = model_path.join("config.json");

    let config_content = std::fs::read_to_string(&config_path).expect("无法读取config.json");
    let bert_config: BertConfig =
        serde_json::from_str(&config_content).expect("无法解析config.json");
    println!("   配置文件加载成功");
    println!("   隐藏层大小: {}", bert_config.hidden_size);
    println!("   注意力头数: {}", bert_config.num_attention_heads);
    println!("   层数: {}", bert_config.num_hidden_layers);
    println!("   词汇表大小: {}", bert_config.vocab_size);

    println!("\n4. 加载tokenizer...");
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = HfTokenizer::from_file(tokenizer_path.to_string_lossy().as_ref())
        .expect("无法加载tokenizer");
    println!("   Tokenizer加载成功");

    println!("\n5. 加载模型权重...");
    let weights_path = model_path.join("model.safetensors");
    let dtype = DType::F32;

    let vb = unsafe {
        match VarBuilder::from_mmaped_safetensors(&[&weights_path], dtype, &device) {
            Ok(vb) => {
                println!("   Safetensors权重加载成功");
                vb
            }
            Err(e) => {
                println!("   Safetensors加载失败: {}", e);
                return;
            }
        }
    };

    println!("\n6. 创建BERT模型...");
    let model = BertModel::load(vb, &bert_config).expect("无法创建模型");
    println!("   模型创建成功");

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

    let vocab_size = bert_config.vocab_size;
    println!("   词汇表大小: {}", vocab_size);

    let max_id = ids.iter().max().copied().unwrap_or(0);
    let min_id = ids.iter().min().copied().unwrap_or(0);
    println!("   Token ID范围: [{}, {}]", min_id, max_id);

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

    println!("   有效Token数量: {}", valid_ids.len());
    if valid_ids.len() != ids.len() {
        println!("   ⚠️ 过滤掉 {} 个特殊token", ids.len() - valid_ids.len());
    }

    let final_ids;
    let final_mask;

    if valid_ids.is_empty() {
        println!("   ⚠️ 没有有效token，使用简单文本重试...");
        let simple_encoding = tokenizer.encode("hello", false).expect("无法编码简单文本");
        let simple_ids: Vec<u32> = simple_encoding
            .get_ids()
            .iter()
            .map(|&x| x as u32)
            .collect();
        let simple_mask: Vec<u32> = simple_encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as u32)
            .collect();
        let simple_max_id = simple_ids.iter().max().copied().unwrap_or(0);
        println!("   简单文本Token ID范围: [0, {}]", simple_max_id);
        println!("   简单文本Tokens: {:?}", simple_encoding.get_tokens());
        final_ids = simple_ids;
        final_mask = simple_mask;
    } else {
        let valid_max_id = valid_ids.iter().max().copied().unwrap_or(0);
        println!("   有效Token ID范围: [0, {}]", valid_max_id);
        final_ids = valid_ids;
        final_mask = valid_mask;
    }

    println!("\n8. 创建输入张量 (同步模式)...");
    let token_ids = Tensor::new(final_ids.as_slice(), &device)
        .expect("无法创建token_ids张量")
        .unsqueeze(0)
        .expect("无法添加batch维度");
    let attention_mask_tensor = Tensor::new(final_mask.as_slice(), &device)
        .expect("无法创建attention_mask张量")
        .unsqueeze(0)
        .expect("无法添加batch维度");
    println!("   token_ids shape: {:?}", token_ids.shape());
    println!(
        "   attention_mask shape: {:?}",
        attention_mask_tensor.shape()
    );

    println!("\n9. 执行前向传播...");
    println!("   ⚠️ 正在执行GPU推理，这可能需要几秒钟...");

    let start = Instant::now();
    let embeddings = model
        .forward(&token_ids, &attention_mask_tensor, None)
        .expect("前向传播失败");
    let inference_time = start.elapsed();

    println!("   推理耗时: {:?}", inference_time);
    println!("   输出形状: {:?}", embeddings.shape());

    println!("\n10. 提取CLS嵌入...");
    let cls_embedding = embeddings
        .get(0)
        .expect("无法获取batch 0")
        .get(0)
        .expect("无法获取CLS token");
    let embedding_vec = cls_embedding.to_vec1::<f32>().expect("无法转换为Vec<f32>");
    println!("   CLS嵌入维度: {}", embedding_vec.len());

    let norm: f32 = embedding_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("   L2范数: {:.4}", norm);

    println!("\n=== GPU测试成功完成 ===");
    println!("\n✅ GPU推理工作正常!");
    println!("   - 设备: NVIDIA CUDA");
    println!("   - 推理时间: {:?}", inference_time);
    println!("   - 输出维度: {}", embedding_vec.len());
}
