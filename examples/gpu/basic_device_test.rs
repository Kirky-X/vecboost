use candle_core::{Device, Result as CandleResult};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertConfig, BertModel};

mod utils;

fn main() -> CandleResult<()> {
    println!("🚀 Candle GPU 基础设备测试");
    println!("================================\n");

    let device = match Device::new_cuda(0) {
        Ok(d) => {
            println!("✅ CUDA设备创建成功: {}", utils::get_device_info(&d));
            d
        }
        Err(e) => {
            println!("❌ CUDA设备创建失败: {}", e);
            return Ok(());
        }
    };

    let model_path = match utils::find_model_path(utils::DEFAULT_MODEL_PATHS) {
        Some(path) => {
            println!("✅ 找到模型路径: {}", path);
            path
        }
        None => {
            println!("❌ 未找到模型文件，请检查以下路径:");
            for path in utils::DEFAULT_MODEL_PATHS {
                println!("   - {}", path);
            }
            return Ok(());
        }
    };

    let weights_path = format!("{}/model.safetensors", model_path);
    if !utils::validate_model_path(&weights_path) {
        println!("❌ 模型权重文件不存在: {}", weights_path);
        return Ok(());
    }

    println!("\n📊 加载模型配置...");
    let dtype = candle_core::DType::F16;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights_path], dtype, &device)
            .expect("无法加载Safetensors权重")
    };

    let config_path = format!("{}/config.json", model_path);
    let bert_config: BertConfig =
        serde_json::from_str(&std::fs::read_to_string(config_path).expect("无法读取配置文件"))
            .expect("无法解析配置文件");

    println!("  模型类型: BERT");
    println!("  隐藏层大小: {}", bert_config.hidden_size);
    println!("  注意力头数: {}", bert_config.num_attention_heads);
    println!("  层数: {}", bert_config.num_hidden_layers);

    println!("\n🔧 创建模型...");
    let model = BertModel::load(vb, &bert_config).expect("无法创建模型");
    println!("✅ 模型创建成功");

    println!("\n📝 测试文本编码...");
    let test_text = utils::get_single_test_text();
    println!("  文本: \"{}\"", test_text);

    println!("\n✅ Candle GPU 基础设备测试完成");
    Ok(())
}
