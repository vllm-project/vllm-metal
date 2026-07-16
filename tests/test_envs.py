# SPDX-License-Identifier: Apache-2.0
"""Tests for boolean environment variable parsing in ``vllm_metal.envs``."""

import pytest

import vllm_metal.envs as envs

# The six boolean env vars and their documented defaults.
BOOL_ENV_DEFAULTS = {
    "VLLM_METAL_USE_MLX": True,
    "VLLM_METAL_DEBUG": False,
    "VLLM_METAL_USE_PAGED_ATTENTION": True,
    "VLLM_METAL_GDN_LAZY_KERNELS": True,
    "VLLM_METAL_MLA_KERNEL": False,
    "VLLM_METAL_BUILD_FROM_SOURCE": False,
}

TRUE_SPELLINGS = [
    "1",
    "true",
    "True",
    "TRUE",
    "yes",
    "YES",
    "on",
    "On",
    " true ",
    "\ton\t",
]

FALSE_SPELLINGS = [
    "0",
    "false",
    "False",
    "FALSE",
    "no",
    "NO",
    "off",
    "Off",
    " false ",
]

INVALID_VALUES = ["enabled", "disabled", "2", "-1", "tru", "yess", "t", "y"]


class TestBoolEnvVars:
    """Boolean ``VLLM_METAL_*`` vars accept 1/0, true/false, yes/no, on/off."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Isolate each test from ambient VLLM_METAL_* environment."""
        for var in envs.environment_variables:
            monkeypatch.delenv(var, raising=False)

    def test_all_bool_vars_registered(self) -> None:
        assert set(BOOL_ENV_DEFAULTS) <= set(envs.environment_variables)

    @pytest.mark.parametrize("name", sorted(BOOL_ENV_DEFAULTS))
    @pytest.mark.parametrize("value", TRUE_SPELLINGS)
    def test_true_spellings(self, monkeypatch, name: str, value: str) -> None:
        monkeypatch.setenv(name, value)
        assert getattr(envs, name) is True

    @pytest.mark.parametrize("name", sorted(BOOL_ENV_DEFAULTS))
    @pytest.mark.parametrize("value", FALSE_SPELLINGS)
    def test_false_spellings(self, monkeypatch, name: str, value: str) -> None:
        monkeypatch.setenv(name, value)
        assert getattr(envs, name) is False

    @pytest.mark.parametrize("name,default", sorted(BOOL_ENV_DEFAULTS.items()))
    def test_unset_uses_default(self, name: str, default: bool) -> None:
        assert getattr(envs, name) is default

    @pytest.mark.parametrize("name,default", sorted(BOOL_ENV_DEFAULTS.items()))
    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_uses_default(
        self, monkeypatch, name: str, default: bool, value: str
    ) -> None:
        monkeypatch.setenv(name, value)
        assert getattr(envs, name) is default

    @pytest.mark.parametrize("name", sorted(BOOL_ENV_DEFAULTS))
    @pytest.mark.parametrize("value", INVALID_VALUES)
    def test_invalid_value_raises(self, monkeypatch, name: str, value: str) -> None:
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            getattr(envs, name)

    def test_error_message_lists_accepted_values(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_DEBUG", "enabled")
        with pytest.raises(ValueError) as excinfo:
            _ = envs.VLLM_METAL_DEBUG
        message = str(excinfo.value)
        assert "VLLM_METAL_DEBUG='enabled'" in message
        assert "1/0, true/false, yes/no, on/off" in message
