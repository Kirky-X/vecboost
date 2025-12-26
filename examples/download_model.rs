const MODEL_ID: &str = "BAAI/bge-m3";

#[tokio::main]
async fn main() {
    println!("=== VecBoost 模型下载工具 ===\n");

    println!("下载模型文件: {}", MODEL_ID);

    let api = hf_hub::api::sync::Api::new().expect("Failed to create HF API");
    let repo = api.repo(hf_hub::Repo::new(
        MODEL_ID.to_string(),
        hf_hub::RepoType::Model,
    ));

    let files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "pytorch_model.bin",
        "model.safetensors",
        "config_sentence_transformers.json",
    ];

    for file in &files {
        match repo.get(file) {
            Ok(path) => {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                println!(
                    "  ✓ {} ({:.1} MB) at {:?}",
                    file,
                    size as f64 / 1024.0 / 1024.0,
                    path
                );
            }
            Err(e) => {
                println!("  ✗ {} - 错误: {}", file, e);
            }
        }
    }

    println!("\n模型下载完成！");
    println!("请使用包含所有必需文件的snapshot路径。");
}
