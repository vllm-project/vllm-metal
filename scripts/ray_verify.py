#!/usr/bin/env python3
"""
Ray verification script for vLLM-Metal.

This script helps verify that Ray is properly configured for use with vLLM-Metal.
"""

import platform


def check_platform():
    """Check if we're on Apple Silicon Mac."""
    if platform.system() != "Darwin":
        print("❌ This script is designed for macOS only")
        return False

    if platform.machine() != "arm64":
        print("❌ This script is designed for Apple Silicon (arm64) only")
        return False

    print("✅ Running on Apple Silicon Mac")
    return True


def check_mlx():
    """Check if MLX is available."""
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            print("✅ MLX Metal is available")
            return True
        else:
            print("❌ MLX Metal is not available")
            return False
    except ImportError:
        print("❌ MLX not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking MLX: {e}")
        return False


def check_ray():
    """Check if Ray is installed and available."""
    try:
        import importlib.util

        if importlib.util.find_spec("ray") is not None:
            print("✅ Ray is installed")
            return True
        else:
            print("❌ Ray not installed. Install with: pip install ray>=2.0.0")
            return False
    except ImportError:
        print("❌ Ray not installed. Install with: pip install ray>=2.0.0")
        return False


def check_ray_cluster():
    """Check if Ray cluster is running."""
    try:
        import ray

        if ray.is_initialized():
            print("⚠️  Ray is already initialized in this process")
            return True

        # Try to connect to an existing cluster
        try:
            ray.init(address="auto", ignore_reinit_error=True)
            nodes = ray.nodes()
            print(f"✅ Ray cluster found with {len(nodes)} node(s)")

            # Print node addresses to help with IP mismatch issues
            for i, node in enumerate(nodes):
                addr = node.get("NodeManagerAddress", "unknown")
                print(f"  Node {i}: {addr}")

            ray.shutdown()
            return True
        except Exception as e:
            print(f"⚠️  No Ray cluster found or connection failed: {e}")
            print(
                "   You may need to start Ray: ray start --head "
                "--node-ip-address=127.0.0.1 --port=6379"
            )
            return False
    except ImportError:
        return False


def check_vllm_metal():
    """Check if vLLM-Metal is available."""
    try:
        from vllm_metal.platform import MetalPlatform

        if MetalPlatform.is_available():
            print("✅ vLLM Metal platform is available")
            print(f"  Ray device key: {MetalPlatform.ray_device_key}")
            return True
        else:
            print("❌ vLLM Metal platform is not available")
            return False
    except ImportError as e:
        print(f"❌ Error importing vLLM Metal: {e}")
        return False


def main():
    print("vLLM-Metal Ray Verification Script")
    print("=" * 40)

    checks = [
        ("Platform Check", check_platform),
        ("MLX Check", check_mlx),
        ("Ray Installation Check", check_ray),
        ("vLLM-Metal Check", check_vllm_metal),
        ("Ray Cluster Check", check_ray_cluster),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        result = check_func()
        results.append((name, result))

    print("\n" + "=" * 40)
    print("Summary:")
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")

    # Provide recommendations based on results
    print("\nRecommendations:")
    if not results[1][1]:  # MLX check failed
        print("  - Install MLX: pip install mlx mlx-lm")

    if not results[2][1]:  # Ray check failed
        print("  - Install Ray: pip install ray>=2.0.0")

    if not results[4][1]:  # Ray cluster check failed
        print("  - Start Ray with explicit IP (critical for Metal):")
        print("    ray stop -f")
        print(
            "    ray start --head --node-ip-address=127.0.0.1 "
            "--port=6379 --disable-usage-stats"
        )
        print("    export VLLM_HOST_IP=127.0.0.1")

    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All checks passed! Ray should work with vLLM-Metal.")
    else:
        print(
            "\n⚠️  Some checks failed. Please address the issues above "
            "before using Ray with vLLM-Metal."
        )
        print(
            "   Note: Ray integration with Metal is experimental and "
            "may require additional configuration."
        )


if __name__ == "__main__":
    main()
