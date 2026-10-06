fn main() {
    // 嵌入 torch 后端：libpython3.11 位于 uv 独立 Python 目录，写入 rpath
    // 使二进制不依赖调用方 shell 的 LD_LIBRARY_PATH
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,/home/dev/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/lib"
    );
    println!("cargo:rerun-if-changed=build.rs");
}
