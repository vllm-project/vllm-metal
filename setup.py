# SPDX-License-Identifier: Apache-2.0
"""Package prebuilt native artifacts without invoking a compiler."""

import sysconfig

from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # The prebuilt extension requires CPython and platform-specific tags.
        return True


setup(
    distclass=BinaryDistribution,
    package_data={
        "vllm_metal.pytorch_backend": ["*.mm"],
        "vllm_metal.metal": [
            f"_paged_ops{sysconfig.get_config_var('EXT_SUFFIX')}",
            "_paged_ops.mlx-version",
            "*.metallib",
            "*.cpp",
            "kernels_v2/*.metal",
        ],
    },
    options={"bdist_wheel": {"plat_name": "macosx_15_0_arm64"}},
)
