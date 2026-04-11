# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import vllm.envs

import vllm_metal as vm
from vllm_metal.envs import environment_variables as metal_env_vars


def test_register_merges_metal_env_vars_into_vllm() -> None:
    vm._register()

    missing = [k for k in metal_env_vars if k not in vllm.envs.environment_variables]
    assert not missing, f"metal env vars not registered with vllm: {missing}"
