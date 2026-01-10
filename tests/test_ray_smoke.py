"""
Smoke tests for Ray integration with vLLM-Metal.

These tests verify basic Ray functionality with Metal platform.
Note: These require manual Ray cluster setup due to the IP address
and resource allocation issues.
"""

import sys

import pytest

# Only run on Apple Silicon Macs
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not sys.maxsize > 2**32,  # Not macOS or not 64-bit
    reason="Metal platform only available on macOS with 64-bit",
)


def test_ray_metal_platform_available():
    """Test that Metal platform is available and can be detected."""
    try:
        from vllm_metal.platform import MetalPlatform

        assert MetalPlatform.is_available(), (
            "Metal platform should be available on Apple Silicon"
        )
    except ImportError:
        pytest.skip("vLLM Metal platform not available")


def test_ray_device_key_exists():
    """Test that ray_device_key is properly defined."""
    from vllm_metal.platform import MetalPlatform

    assert hasattr(MetalPlatform, "ray_device_key")
    assert MetalPlatform.ray_device_key is not None
    # Verify the ray_device_key is set to "METAL" to avoid CPU resource conflicts
    assert MetalPlatform.ray_device_key == "METAL", (
        f"Expected 'METAL', got '{MetalPlatform.ray_device_key}'"
    )


@pytest.mark.skip(
    reason="Ray integration requires manual cluster setup "
    "and explicit IP/resource configuration"
)
def test_ray_single_node_basic():
    """Basic smoke test for Ray with Metal - requires manual setup."""
    # This test is skipped by default due to the complex setup requirements
    # To run this test manually:
    # 1. Start Ray with explicit IP and custom resource: ray start --head
    #    --node-ip-address=127.0.0.1 --port=6379 --resources='{"METAL": 1}'
    # 2. Set environment: export VLLM_HOST_IP=127.0.0.1

    try:
        import ray

        # Check if Ray is already initialized (manually)
        if not ray.is_initialized():
            pytest.skip(
                "Ray must be manually started with correct "
                "IP and resource configuration"
            )

        # Test basic platform detection with Ray
        from vllm_metal.platform import MetalPlatform

        # Verify platform can be detected properly
        assert MetalPlatform.is_available()

        # Test that ray_device_key exists and is properly set
        ray_device_key = MetalPlatform.ray_device_key
        assert ray_device_key is not None
        assert ray_device_key == "METAL", f"Expected 'METAL', got '{ray_device_key}'"

        print(f"Metal platform Ray device key: {ray_device_key}")

        # Test that Ray runtime context can be accessed (though custom resources
        # won't appear in accelerator IDs)
        try:
            ctx = ray.get_runtime_context()
            print(f"Ray job ID: {ctx.job_id}")
        except Exception as e:
            print(f"Could not access Ray runtime context: {e}")

    except ImportError:
        pytest.skip("Ray not installed or not available")


def test_ray_configuration_requirements():
    """Document the required Ray configuration for Metal to work."""
    # This test documents the manual configuration steps needed
    config_steps = [
        "1. Stop any existing Ray cluster: ray stop -f",
        (
            "2. Start Ray with explicit IP and custom resource: ray start --head "
            "--node-ip-address=127.0.0.1 --port=6379 --disable-usage-stats "
            "--resources='{\"METAL\": 1}'"
        ),
        "3. Set VLLM_HOST_IP to match Ray IP: export VLLM_HOST_IP=127.0.0.1",
        (
            "4. Run vLLM with Ray backend: vllm serve <model> "
            "--distributed-executor-backend ray"
        ),
        (
            "5. Note: Due to upstream vLLM limitations, custom resources "
            "like 'METAL' may not appear in accelerator IDs"
        ),
    ]

    # Print the required steps for reference
    print("\nRequired Ray configuration for Metal:")
    for step in config_steps:
        print(f"  {step}")

    # This test always passes but serves as documentation
    assert True


def test_ray_experimental_warning():
    """Test that experimental warning is properly logged when Ray is used."""
    # This test verifies that the experimental warning is shown
    from unittest.mock import patch

    # Capture log messages to verify warning is issued
    with patch("logging.Logger.warning") as mock_warning:
        from vllm_metal.platform import MetalPlatform

        # Create a mock VllmConfig to test the check_and_update_config method
        class MockParallelConfig:
            def __init__(self):
                self.distributed_executor_backend = "ray"
                self.worker_cls = "auto"
                self.disable_custom_all_reduce = False

        class MockCacheConfig:
            def __init__(self):
                self.block_size = None

        class MockModelConfig:
            def __init__(self):
                self.disable_cascade_attn = False

        class MockVllmConfig:
            def __init__(self):
                self.parallel_config = MockParallelConfig()
                self.cache_config = MockCacheConfig()
                self.model_config = MockModelConfig()

        vllm_config = MockVllmConfig()

        # Call the method that should trigger the warning
        MetalPlatform.check_and_update_config(vllm_config)

        # Verify that warning was called with experimental notice
        warning_calls = [
            call
            for call in mock_warning.call_args_list
            if "experimental" in str(call).lower()
        ]
        assert len(warning_calls) > 0, (
            "Experimental warning should be logged when Ray is used"
        )


if __name__ == "__main__":
    # Run basic tests that don't require Ray cluster
    test_ray_metal_platform_available()
    test_ray_device_key_exists()
    test_ray_configuration_requirements()
    test_ray_experimental_warning()
    print(
        "Basic smoke tests passed (Ray cluster tests skipped due to setup requirements)"
    )
