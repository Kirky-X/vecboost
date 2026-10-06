// Prep-phase smoke test: verifies the onnxruntime dylib loads via ort
// `load-dynamic` and a session can be created from the VAD model. Run:
//   cargo run --example ort_dylib_smoke
// (dylib path and model path resolved below; override via ORT_DYLIB_ARGS not
// needed — edit the constants if the layout moves)
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dylib = concat!(env!("CARGO_MANIFEST_DIR"), "/../3rdparty/libonnxruntime.so.1.28.2");
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/models/silero_vad.onnx");

    ort::init_from(dylib)
        .expect("failed to load onnxruntime dylib")
        .with_name("ort_dylib_smoke")
        .commit();

    let session = ort::session::Session::builder()?.commit_from_file(model)?;
    let n_in = session.inputs().len();
    let n_out = session.outputs().len();
    println!("DYLIB_OK dylib={dylib} model={model} inputs={n_in} outputs={n_out}");
    Ok(())
}
