use std::{env, fs, path::PathBuf};

fn main() {
    const SHADER_PATH: &str = "shaders/vector_add.wgsl";

    println!("cargo:rerun-if-changed={SHADER_PATH}");

    let source = fs::read_to_string(SHADER_PATH).expect("failed to read vector_add WGSL");
    let module = naga::front::wgsl::parse_str(&source).expect("failed to parse vector_add WGSL");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("failed to validate vector_add WGSL");
    let words = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .expect("failed to compile vector_add WGSL to SPIR-V");

    let output =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set")).join("vector_add.spv");
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(output, bytes).expect("failed to write vector_add SPIR-V");
}
