fn main() {
    println!("cargo:rerun-if-changed=src/hipblaslt_shim.cpp");
    println!("cargo:rerun-if-changed=src/fused_add_rms_norm.hip");
    println!("cargo:rerun-if-changed=src/quantized_linear.hip");
    println!("cargo:rerun-if-changed=src/rms_norm.hip");
    println!("cargo:rerun-if-changed=src/rope.hip");
    println!("cargo:rerun-if-env-changed=MOTE_HIP_ARCH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=AR");

    if std::env::var_os("CARGO_FEATURE_ROCM").is_none() {
        return;
    }

    cc::Build::new()
        .cpp(true)
        .file("src/hipblaslt_shim.cpp")
        .define("__HIP_PLATFORM_AMD__", None)
        .define("ROCM_USE_FLOAT16", None)
        .opt_level(2)
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-Wpedantic")
        .compile("mote_hipblaslt_shim");

    compile_hip_kernels();

    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblaslt");
}

/// Compiles the `hipcc` kernels (`src/fused_add_rms_norm.hip`,
/// `src/quantized_linear.hip`, `src/rms_norm.hip`, `src/rope.hip`) and
/// archives them into one static library for linking.
///
/// The offload architecture defaults to `--offload-arch=native` (the building
/// machine's GPU); setting `MOTE_HIP_ARCH` overrides it, e.g.
/// `MOTE_HIP_ARCH=gfx1100`. `HIPCC` and `AR` select the tools.
fn compile_hip_kernels() {
    use std::{env, path::PathBuf, process::Command};

    const SOURCES: &[&str] = &[
        "src/fused_add_rms_norm.hip",
        "src/quantized_linear.hip",
        "src/rms_norm.hip",
        "src/rope.hip",
    ];

    let arch = env::var("MOTE_HIP_ARCH").unwrap_or_else(|_| "native".to_owned());
    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR for build scripts"));
    let objects: Vec<PathBuf> = SOURCES
        .iter()
        .map(|source| out_dir.join(format!("{}.o", source.rsplit('/').next().unwrap_or(source))))
        .collect();
    let archive = out_dir.join("libmote_hip_kernels.a");

    let hipcc = env::var_os("HIPCC").unwrap_or_else(|| "hipcc".into());
    for (source, object) in SOURCES.iter().zip(&objects) {
        let status = Command::new(&hipcc)
            .arg("-c")
            .arg(source)
            .arg(format!("--offload-arch={arch}"))
            .arg("-O3")
            .arg("-fPIC")
            .arg("-o")
            .arg(object)
            .status()
            .unwrap_or_else(|error| panic!("failed to spawn {hipcc:?} for {source}: {error}"));
        assert!(
            status.success(),
            "hipcc failed for {source} with --offload-arch={arch}",
        );
    }

    // Rebuild the archive from scratch so sources dropped from SOURCES cannot
    // leave stale members behind.
    let _ = std::fs::remove_file(&archive);

    let ar = env::var_os("AR").unwrap_or_else(|| "ar".into());
    let status = Command::new(&ar)
        .arg("rcs")
        .arg(&archive)
        .args(&objects)
        .status()
        .unwrap_or_else(|error| panic!("failed to spawn {ar:?}: {error}"));
    assert!(status.success(), "ar failed for {archive:?}");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mote_hip_kernels");
}
