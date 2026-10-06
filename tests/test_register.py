# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
import os
import re
from collections.abc import Iterator

import pytest
import vllm.envs
import vllm.logger
from vllm.config import LoggingConfig
from vllm.logger import configure_logging

import vllm_metal as vm
from vllm_metal.envs import environment_variables as metal_env_vars
from vllm_metal.platform import MetalPlatform

_PROBE = "metal logging probe"


@pytest.fixture
def unconfigured_loggers(monkeypatch) -> Iterator[None]:
    """Reset vllm and vllm_metal logging to its state before vLLM configures it."""
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    monkeypatch.setenv("VLLM_LOGGING_STREAM", "ext://sys.stdout")
    monkeypatch.setenv("VLLM_LOGGING_COLOR", "0")
    monkeypatch.setattr(vm, "_mirrored_handlers", [])
    monkeypatch.setattr(vllm.logger, "_last_configured_logging_config", None)
    # configure_logging also quiets httpx/httpx2 when it applies INFO.
    loggers = [
        logging.getLogger(name) for name in ("vllm", "vllm_metal", "httpx", "httpx2")
    ]
    saved = [(lg, lg.handlers[:], lg.level, lg.propagate) for lg in loggers]
    record_factory = logging.getLogRecordFactory()
    for lg in loggers:
        lg.handlers = []
        lg.setLevel(logging.NOTSET)
        lg.propagate = True
    yield
    for lg, handlers, level, propagate in saved:
        lg.handlers = handlers
        lg.setLevel(level)
        lg.propagate = propagate
    logging.setLogRecordFactory(record_factory)


def _logging_config(log_level: str) -> LoggingConfig:
    return LoggingConfig(
        log_level=log_level, configure_logging=True, pylogging_config_file=None
    )


def test_register_merges_metal_env_vars_into_vllm() -> None:
    vm._register()

    missing = [k for k in metal_env_vars if k not in vllm.envs.environment_variables]
    assert not missing, f"metal env vars not registered with vllm: {missing}"


def test_register_pins_v1_model_runner_when_metal_is_selected(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(MetalPlatform, "is_available", classmethod(lambda cls: True))

    assert vm._register() == "vllm_metal.platform.MetalPlatform"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_register_leaves_model_runner_alone_when_metal_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(MetalPlatform, "is_available", classmethod(lambda cls: False))

    assert vm._register() is None
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ


def test_vllm_logging_config_reaches_metal_loggers(
    unconfigured_loggers, capsys
) -> None:
    metal_logger = logging.getLogger("vllm_metal.test_register")

    configure_logging(_logging_config("DEBUG"))
    metal_logger.debug(_PROBE)
    out = capsys.readouterr().out

    assert re.search(rf" DEBUG .* \[test_register\.py:\d+\] {_PROBE}$", out, re.M)
    assert logging.getLogger("vllm_metal").propagate is False


def test_vllm_logging_reconfiguration_reaches_metal_loggers(
    unconfigured_loggers, capsys
) -> None:
    metal_logger = logging.getLogger("vllm_metal.test_register")
    configure_logging(_logging_config("INFO"))

    configure_logging(_logging_config("DEBUG"))
    metal_logger.debug(_PROBE)
    metal_logger.info(_PROBE)
    out = capsys.readouterr().out

    assert re.findall(rf" (DEBUG|INFO) .* {_PROBE}$", out, re.M) == ["DEBUG", "INFO"]


def test_vllm_logging_level_reaches_user_attached_metal_handlers(
    unconfigured_loggers,
) -> None:
    metal_logger = logging.getLogger("vllm_metal.test_register")
    stream = io.StringIO()
    logging.getLogger("vllm_metal").addHandler(logging.StreamHandler(stream))

    configure_logging(_logging_config("INFO"))
    metal_logger.info(_PROBE)

    assert stream.getvalue() == f"{_PROBE}\n"


def test_metal_logging_follows_the_host_when_vllm_does_not_configure(
    unconfigured_loggers, caplog
) -> None:
    metal_logger = logging.getLogger("vllm_metal.test_register")
    vm._register()
    configure_logging(
        LoggingConfig(configure_logging=False, pylogging_config_file=None)
    )
    caplog.set_level(logging.DEBUG)

    metal_logger.debug(_PROBE)

    messages = [r.getMessage() for r in caplog.records if r.name == metal_logger.name]
    assert messages == [_PROBE]
