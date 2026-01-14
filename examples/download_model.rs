use std::env;
use std::path::PathBuf;

const DEFAULT_TEST_MODEL: &str = "BAAI/bge-small-en-v1.5";
const DEFAULT_FULL_MODEL: &str = "BAAI/bge-m3";

// 测试用的小模型文件列表
const SMALL_MODEL_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin",
];

// 完整模型文件列表
const FULL_MODEL_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin",
    "model.safetensors",
    "config_sentence_transformers.json",
];

fn print_usage() {
    println!(
        r#"VecBoost 模型下载工具

用法: cargo run --example download_model [选项]

选项:
  --model <MODEL_ID>    指定要下载的模型 ID (默认: {})"#,
        DEFAULT_TEST_MODEL
    );
    println!(
        r#"  --small             下载测试用的小模型 (BAAI/bge-small-en-v1.5)
  --full                下载完整模型 (BAAI/bge-m3)
  --list                列出支持的模型
  --help, -h            显示此帮助信息

示例:
  cargo run --example download_model --small
  cargo run --example download_model --model sentence-transformers/all-MiniLM-L6-v2
"#
    );
}

fn list_supported_models() {
    println!("支持的测试模型:");
    println!();
    println!("  模型名称                              维度  大小估计");
    println!("  ------------------------------------ ----- -----------");
    println!("  BAAI/bge-small-en-v1.5               384   ~90 MB");
    println!("  BAAI/bge-base-en-v1.5               768   ~400 MB");
    println!("  BAAI/bge-large-en-v1.5              1024  ~1.4 GB");
    println!("  BAAI/bge-m3                         1024  ~2.2 GB");
    println!("  sentence-transformers/all-MiniLM-L6-v2 384  ~90 MB");
}

fn get_model_files(model_id: &str) -> &'static [&'static str] {
    match model_id {
        "BAAI/bge-small-en-v1.5" | "sentence-transformers/all-MiniLM-L6-v2" => &SMALL_MODEL_FILES,
        _ => &FULL_MODEL_FILES,
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.1} GB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "--help" | "-h" => {
                print_usage();
                return;
            }
            "--list" => {
                list_supported_models();
                return;
            }
            "--small" => {
                println!("=== VecBoost 模型下载工具 ===\n");
                println!("下载测试用小模型: {}", DEFAULT_TEST_MODEL);
                download_model(DEFAULT_TEST_MODEL, &SMALL_MODEL_FILES).await;
                return;
            }
            "--full" => {
                println!("=== VecBoost 模型下载工具 ===\n");
                println!("下载完整模型: {}", DEFAULT_FULL_MODEL);
                download_model(DEFAULT_FULL_MODEL, &FULL_MODEL_FILES).await;
                return;
            }
            "--model" if args.len() > 2 => {
                let model_id = &args[2];
                let files = get_model_files(model_id);
                println!("=== VecBoost 模型下载工具 ===\n");
                println!("下载模型: {}", model_id);
                download_model(model_id, files).await;
                return;
            }
            _ => {
                eprintln!("未知参数: {}", args[1]);
                print_usage();
                return;
            }
        }
    }

    // 默认行为：下载测试用小模型
    println!("=== VecBoost 模型下载工具 ===\n");
    println!("下载测试用小模型: {}", DEFAULT_TEST_MODEL);
    println!("(使用 --full 下载完整模型，--list 查看支持的模型)\n");
    download_model(DEFAULT_TEST_MODEL, &SMALL_MODEL_FILES).await;
}

async fn download_model(model_id: &str, files: &[&str]) {
    println!();
    println!("目标目录: models/{}", model_id.replace('/', "-"));
    println!();

    // 创建输出目录
    let output_dir = PathBuf::from("models").join(model_id.replace('/', "-"));
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("创建输出目录失败: {}", e);
        return;
    }

    println!("开始下载模型文件...\n");

    let api = hf_hub::api::sync::Api::new().expect("Failed to create HF API");
    let repo = api.repo(hf_hub::Repo::new(
        model_id.to_string(),
        hf_hub::RepoType::Model,
    ));

    let mut success_count = 0;
    let mut total_size: u64 = 0;

    for file in files {
        match repo.get(file) {
            Ok(path) => {
                let path_ref: &std::path::Path = path.as_ref();
                let size = std::fs::metadata(path_ref).map(|m| m.len()).unwrap_or(0);
                println!("  ✓ {} ({})", file, format_size(size));
                total_size += size;

                // 复制到输出目录
                if let Some(filename) = path_ref.file_name() {
                    let dest = output_dir.join(filename);
                    if let Err(e) = std::fs::copy(path_ref, &dest) {
                        println!("    复制到 {} 失败: {}", dest.display(), e);
                    } else {
                        success_count += 1;
                    }
                }
            }
            Err(e) => {
                println!("  ✗ {} - 错误: {}", file, e);
            }
        }
    }

    println!();
    println!("下载完成!");
    println!("  成功: {}/{}", success_count, files.len());
    println!("  总大小: {}", format_size(total_size));
    println!();
    println!("模型文件位于: {:?}", output_dir);
    println!();
    println!("使用方法:");
    println!("  cargo run --release -- --model-path {:?}", output_dir);
}
