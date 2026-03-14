// SPDX-License-Identifier: Apache-2.0
//! Build script: compile paged_ops_bridge.cpp (C ABI shim over MLX C++) and
//! link the resulting static library into the Rust extension.
//!
//! Paths are resolved by shelling out to the Python that maturin configured
//! (available as the `PYTHON_SYS_EXECUTABLE` env var that maturin injects, or
//! `python3` as a fallback).

use std::path::PathBuf;
use std::process::Command;

fn python_path(script: &str) -> PathBuf {
    let python = std::env::var("PYTHON_SYS_EXECUTABLE")
        .unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python)
        .args(["-c", script])
        .output()
        .unwrap_or_else(|e| panic!("python3 invocation failed: {e}"));
    assert!(
        out.status.success(),
        "Python script failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    PathBuf::from(String::from_utf8(out.stdout).unwrap().trim().to_string())
}

fn main() {
    // Only needed on macOS (Apple Silicon).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    // -----------------------------------------------------------------------
    // Locate dependencies via Python
    // -----------------------------------------------------------------------

    let mlx_root = python_path(
        "import mlx, pathlib; print(pathlib.Path(mlx.__file__).parent)",
    );
    let mlx_include = mlx_root.join("include");
    let mlx_lib     = mlx_root.join("lib");

    let nb_root = python_path(
        "import nanobind, pathlib; print(pathlib.Path(nanobind.__file__).parent)",
    );
    let nb_include    = nb_root.join("include");
    let nb_src        = nb_root.join("src");
    let nb_robin_map  = nb_root.join("ext").join("robin_map").join("include");
    let nb_combined   = nb_src.join("nb_combined.cpp");

    let metal_cpp = mlx_include.join("metal_cpp");

    let py_include = python_path(
        "import sysconfig; print(sysconfig.get_paths()['include'])",
    );

    // -----------------------------------------------------------------------
    // Compile the C ABI bridge (C++ file, C-symbol output)
    // -----------------------------------------------------------------------

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        // Python headers
        .include(&py_include)
        // nanobind
        .include(&nb_include)
        .include(&nb_src)
        .include(&nb_robin_map)
        // MLX
        .include(&mlx_include)
        .include(&metal_cpp)
        // Defines expected by MLX / nanobind on Darwin
        .define("_METAL_", None)
        .define("ACCELERATE_NEW_LAPACK", None)
        // Suppress warnings from third-party headers
        .flag("-Wno-unused-parameter")
        .flag("-Wno-deprecated-declarations")
        // nanobind combined source
        .file(&nb_combined)
        // Our bridge
        .file("src/paged_ops_bridge.cpp")
        .compile("paged_ops_bridge");

    // -----------------------------------------------------------------------
    // Link against libmlx and system frameworks
    // -----------------------------------------------------------------------

    println!("cargo:rustc-link-search=native={}", mlx_lib.display());
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Embed rpath so the .so finds libmlx.dylib at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", mlx_lib.display());

    // -----------------------------------------------------------------------
    // Rebuild triggers
    // -----------------------------------------------------------------------

    println!("cargo:rerun-if-changed=src/paged_ops_bridge.cpp");
    println!("cargo:rerun-if-changed=src/paged_ops_bridge.h");
}
