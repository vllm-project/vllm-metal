require "uri"

class VllmMetal < Formula
  desc "LLM inference server for Apple Silicon using vLLM"
  homepage "https://github.com/vllm-project/vllm-metal"
  # Pin a stable release; old development release assets are deleted.
  url "https://github.com/vllm-project/vllm-metal/releases/download/v0.30.0/vllm_metal-0.30.0-cp312-cp312-macosx_15_0_arm64.whl",
      using: :nounzip
  version "0.30.0"
  sha256 "3955de8b2c75b633ef889b70fa469a4db661a0c5bb2f78e36272d68b16da9353"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on macos: :sequoia
  depends_on "python@3.12"

  # Wheels use relative library IDs. Expanding them to Cellar paths can exceed
  # their Mach-O header space and break MLX's native-extension linkage.
  preserve_rpath

  # Match .github/vllm-release-tag.commit at the plugin tag when updating.
  # The macOS CPU wheel supplies vLLM; this plugin provides Metal execution.
  resource "vllm" do
    url "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl"
    sha256 "fd9adfd566a8afa4ecbdf36b04bab10ebf43cd9ca18eca4b0a0bf2013b8de135"
  end

  def install
    system "python3.12", "-m", "venv", libexec

    # Restore the wheel filename: Homebrew's cached download has a hash prefix
    # that pip cannot parse. Resolve both wheels together, including the
    # plugin's Git-pinned mlx-lm dependency and exact MLX version.
    wheel_name = File.basename(URI.parse(resource("vllm").url).path)
    vllm_wheel = buildpath/URI::DEFAULT_PARSER.unescape(wheel_name)
    cp resource("vllm").cached_download, vllm_wheel
    system libexec/"bin/pip", "install", vllm_wheel,
           buildpath/"vllm_metal-#{version}-cp312-cp312-macosx_15_0_arm64.whl"

    # TVM-FFI uses wheel RECORD files to locate its own and xgrammar's native
    # libraries. Save them outside site-packages before Homebrew cleans them.
    site_packages = libexec/"lib/python3.12/site-packages"
    %w[apache_tvm_ffi xgrammar].each do |package|
      record = site_packages.glob("#{package}-*.dist-info/RECORD").fetch(0)
      (pkgshare/"#{package}.record").write record.read
    end

    bin.install_symlink libexec/"bin/vllm"
  end

  def post_install
    site_packages = libexec/"lib/python3.12/site-packages"
    %w[apache_tvm_ffi xgrammar].each do |package|
      metadata = site_packages.glob("#{package}-*.dist-info").fetch(0)
      cp pkgshare/"#{package}.record", metadata/"RECORD"
    end
  end

  def caveats
    <<~EOS
      Start a server with:
        vllm serve <model>

      Python and the vLLM Metal plugin are installed in a private environment.
      No virtual environment activation is needed.
    EOS
  end

  test do
    assert_match resource("vllm").version.to_s, shell_output("#{bin}/vllm --version 2>&1")
    system libexec/"bin/python", "-c", <<~PYTHON
      import importlib.metadata
      import mlx.core as mx
      from vllm.platforms import current_platform
      from vllm_metal.metal import get_ops
      from vllm_metal.platform import MetalPlatform

      assert importlib.metadata.version("vllm-metal") == "#{version}"
      assert isinstance(current_platform, MetalPlatform), current_platform
      # Load the shipped extension and shaders, then execute a native kernel.
      key = mx.ones((1, 1, 64), dtype=mx.float16)
      value = key * 2
      key_cache = mx.zeros((1, 16, 1, 64), dtype=mx.float16)
      value_cache = mx.zeros((1, 16, 1, 64), dtype=mx.float16)
      slots = mx.array([3], dtype=mx.int64)
      keys, values = get_ops().reshape_and_cache(key, value, key_cache, value_cache, slots)
      assert mx.sum(keys).item() == 64
      assert mx.sum(values).item() == 128
    PYTHON
  end
end
