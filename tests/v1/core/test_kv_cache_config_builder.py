# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    get_kv_cache_config_builder,
)
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    get_kv_cache_groups_from_builder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec


def _make_vllm_config() -> MagicMock:
    """Create a minimal mock VllmConfig for builder resolution tests."""
    return MagicMock()


CUSTOM_PATH = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
DEFAULT_PATH = "vllm.v1.core.kv_cache_planning.DefaultKVCacheConfigBuilder"


class CustomBuilder(DefaultKVCacheConfigBuilder):
    """A test builder subclass."""

    pass


class TestPlatformHookResolution:
    """The platform hook selects only the placement builder."""

    def test_default_hook_returns_default_builder(self):
        cfg = _make_vllm_config()
        assert Platform.get_kv_cache_config_builder_cls(cfg) == DEFAULT_PATH


class TestBuilderResolution:
    @patch("vllm.platforms.current_platform")
    @pytest.mark.parametrize(
        ("builder_cls_path", "expected_cls"),
        [
            (None, DefaultKVCacheConfigBuilder),
            (CUSTOM_PATH, CustomBuilder),
        ],
    )
    def test_resolves_hook_selected_builder(
        self, mock_platform, builder_cls_path, expected_cls
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = (
            builder_cls_path or DEFAULT_PATH
        )
        cfg = _make_vllm_config()
        assert type(get_kv_cache_config_builder(cfg)) is expected_cls

    @patch("vllm.platforms.current_platform")
    def test_resolves_fresh_instance_per_call(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert get_kv_cache_config_builder(cfg) is not get_kv_cache_config_builder(cfg)

    def test_public_interface_declares_two_placement_hooks(self):
        # get_kv_cache_configs lives in kv_cache_planning and is Core-owned;
        # the builder only declares the placement hooks.
        assert KVCacheConfigBuilder.__abstractmethods__ == {
            "get_kv_cache_groups",
            "get_kv_cache_config_from_groups",
        }
        assert issubclass(DefaultKVCacheConfigBuilder, KVCacheConfigBuilder)


class TestCoreChecksGrouping:
    """Core checks the grouping a builder produces."""

    _SPEC = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype="float16"
    )

    @pytest.mark.parametrize(
        ("groups", "match"),
        [
            ([KVCacheGroupSpec(["a"], _SPEC)], "miss layers"),
            (
                [KVCacheGroupSpec(["a", "b"], _SPEC), KVCacheGroupSpec(["b"], _SPEC)],
                "overlaps",
            ),
        ],
    )
    def test_platform_grouping_is_checked(self, groups, match):
        class FixedGroupsBuilder(DefaultKVCacheConfigBuilder):
            def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
                return groups

        with pytest.raises(AssertionError, match=match):
            get_kv_cache_groups_from_builder(
                FixedGroupsBuilder(),
                _make_vllm_config(),
                {"a": self._SPEC, "b": self._SPEC},
            )


class TestPlatformCustomPriority:
    """A vendor platform can override the hook to customize priority."""

    def test_platform_builder_wins_over_model_declaration(self):
        class PlatformFirstPlatform(Platform):
            @classmethod
            def get_kv_cache_config_builder_cls(cls, vllm_config):
                return CUSTOM_PATH

        cfg = _make_vllm_config()
        assert PlatformFirstPlatform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH
        with patch("vllm.platforms.current_platform", PlatformFirstPlatform):
            assert isinstance(get_kv_cache_config_builder(cfg), CustomBuilder)
