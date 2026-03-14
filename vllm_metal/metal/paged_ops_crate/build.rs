// SPDX-License-Identifier: Apache-2.0
//! Build script: compiles mlx_ffi.cpp and links against MLX + nanobind.
//!
//! Locates MLX and nanobind via their Python package paths, mirroring the
//! approach in the original build.py JIT compiler.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn python_package_path(package: &str) -> PathBuf {
    let output = Command::new("python3")
        .args([
            "-c",
            &format!(
                "import importlib, pathlib; \
                 m = importlib.import_module('{package}'); \
                 p = getattr(m, '__path__', None); \
                 print(list(p)[0] if p else str(pathlib.Path(m.__file__).parent))"
            ),
        ])
        .output()
        .unwrap_or_else(|e| panic!("Failed to locate Python package '{package}': {e}"));

    assert!(
        output.status.success(),
        "python3 failed to locate '{package}': {}",
        String::from_utf8_lossy(&output.stderr)
    );

    PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
}

fn python_include_path() -> PathBuf {
    let output = Command::new("python3")
        .args(["-c", "import sysconfig; print(sysconfig.get_paths()['include'])"])
        .output()
        .expect("Failed to get Python include path");

    PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
}

fn main() {
    let mlx_path = python_package_path("mlx");
    let nb_path = python_package_path("nanobind");
    let py_include = python_include_path();

    let mlx_include = mlx_path.join("include");
    let mlx_lib = mlx_path.join("lib");
    let metal_cpp = mlx_include.join("metal_cpp");

    let ffi_src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .join("mlx_ffi.cpp");

    let nb_combined = nb_path.join("src").join("nb_combined.cpp");

    // Compile mlx_ffi.cpp + nanobind runtime into a static library
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .flag("-fvisibility=default")
        .define("_METAL_", None)
        .define("ACCELERATE_NEW_LAPACK", None)
        .include(&py_include)
        .include(nb_path.join("include"))
        .include(nb_path.join("src"))
        .include(nb_path.join("ext").join("robin_map").join("include"))
        .include(&mlx_include)
        .include(&metal_cpp)
        .file(&ffi_src)
        .file(&nb_combined)
        .compile("mlx_ffi");

    // Link against MLX shared library
    println!("cargo:rustc-link-search=native={}", mlx_lib.display());
    println!("cargo:rustc-link-lib=dylib=mlx");

    // macOS frameworks
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Runtime library path
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", mlx_lib.display());

    // Rerun if sources change
    println!("cargo:rerun-if-changed={}", ffi_src.display());
    println!("cargo:rerun-if-changed={}", nb_combined.display());
}
