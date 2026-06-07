"""Process-start compatibility patches for the local MI100 vLLM stack."""

from __future__ import annotations

import json
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__name__),
        )
        return f"{module}.{qualname}"
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _patch_dynamo_metrics_config_logging() -> None:
    try:
        import torch._dynamo.config as dynamo_config
        import torch._dynamo.utils as dynamo_utils
    except Exception:
        return

    original = getattr(dynamo_utils, "_get_dynamo_config_for_logging", None)
    if original is None or getattr(original, "_vllm_json_safe", False):
        return

    blocklist = {
        "TYPE_CHECKING",
        "log_file_name",
        "verbose",
        "repro_after",
        "repro_level",
        "repro_forward_only",
        "repro_tolerance",
        "repro_ignore_non_fp",
        "same_two_models_use_fp64",
        "base_dir",
        "debug_dir_root",
        "_save_config_ignore",
        "log_compilation_metrics",
        "inject_BUILD_SET_unimplemented_TESTING_ONLY",
        "_autograd_backward_strict_mode_banned_ops",
        "reorderable_logging_functions",
        "ignore_logger_methods",
        "ignore_logging_functions",
        "traceable_tensor_subclasses",
        "nontraceable_tensor_subclasses",
        "_custom_ops_profile",
    }

    def _get_dynamo_config_for_logging_json_safe() -> str | None:
        try:
            return original()
        except TypeError:
            config_dict = {
                key: _json_safe(value)
                for key, value in dynamo_config.get_config_copy().items()
                if key not in blocklist
            }
            return json.dumps(config_dict, sort_keys=True)

    _get_dynamo_config_for_logging_json_safe._vllm_json_safe = True  # type: ignore[attr-defined]
    dynamo_utils._get_dynamo_config_for_logging = (  # type: ignore[attr-defined]
        _get_dynamo_config_for_logging_json_safe
    )


_patch_dynamo_metrics_config_logging()
