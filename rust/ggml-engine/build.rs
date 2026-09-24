// SPDX-License-Identifier: Apache-2.0
//! Generates FFI bindings for the system ggml (`brew install ggml`) and wires
//! up linking. The ggml prefix is resolved from `GGML_PREFIX`, then
//! `brew --prefix ggml`, then the default Homebrew location.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn ggml_prefix() -> PathBuf {
    if let Ok(p) = env::var("GGML_PREFIX") {
        return PathBuf::from(p);
    }
    if let Ok(out) = Command::new("brew").args(["--prefix", "ggml"]).output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return PathBuf::from(s);
            }
        }
    }
    PathBuf::from("/opt/homebrew/opt/ggml")
}

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_PREFIX");
    println!("cargo:rerun-if-changed=wrapper.h");

    let prefix = ggml_prefix();
    let include = prefix.join("include");
    let lib = prefix.join("lib");
    let libexec = prefix.join("libexec");
    assert!(
        include.join("ggml.h").exists(),
        "ggml headers not found under {} (run `brew install ggml` or set GGML_PREFIX)",
        include.display()
    );

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    // Resolve libggml*.dylib at runtime without DYLD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    // Default directory holding the dynamically loaded backends
    // (libggml-metal.so, libggml-cpu-*.so). Overridable at runtime.
    println!(
        "cargo:rustc-env=GGML_BACKEND_DIR_DEFAULT={}",
        libexec.display()
    );

    // Python symbols are resolved by the host interpreter on macOS.
    pyo3_build_config::add_extension_module_link_args();

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .expect("bindgen failed for ggml headers");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("ggml_bindings.rs");
    bindings
        .write_to_file(out)
        .expect("failed to write bindings");
}
