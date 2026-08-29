use std::{env, path::PathBuf, process::Command};

fn main() {
    const SHADERS: &[(&str, &str)] = &[
        ("shaders/vector_add.slang", "vector_add.spv"),
        ("shaders/matmul.slang", "matmul.spv"),
        ("shaders/matmul_cmma.slang", "matmul_cmma.spv"),
    ];

    println!("cargo:rerun-if-env-changed=SLANGC");

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set"));
    let output_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let slangc = env::var_os("SLANGC").unwrap_or_else(|| "slangc".into());

    for &(shader_path, output_name) in SHADERS {
        println!("cargo:rerun-if-changed={shader_path}");
        compile_shader(
            &slangc,
            &manifest_dir.join(shader_path),
            &output_dir.join(output_name),
        );
    }
}

fn compile_shader(
    slangc: &std::ffi::OsStr,
    shader: &std::path::Path,
    output_path: &std::path::Path,
) {
    let status = Command::new(slangc)
        .arg(shader)
        .args([
            "-entry",
            "main",
            "-stage",
            "compute",
            "-target",
            "spirv",
            "-profile",
            "glsl_460+spirv_1_3",
            "-O3",
            "-o",
        ])
        .arg(output_path)
        .status()
        .unwrap_or_else(|error| {
            panic!(
                "failed to run `{}`: {error}; enter `nix develop` or set SLANGC",
                PathBuf::from(&slangc).display()
            )
        });

    assert!(
        status.success(),
        "Slang failed to compile {}",
        shader.display()
    );
}
