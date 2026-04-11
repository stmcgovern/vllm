# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Platform.inference_mode() dispatch."""

import torch

from vllm.platforms.interface import Platform, PlatformEnum


class NoGradPlatform(Platform):
    """Simulates a backend that doesn't support torch.inference_mode."""

    _enum = PlatformEnum.OOT
    device_name = "no_grad_device"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()


def test_default_inference_mode_disables_grad():
    ctx = Platform.inference_mode()
    with ctx:
        t = torch.randn(2, 2)
        assert not t.requires_grad


def test_default_inference_mode_is_inference_mode():
    ctx = Platform.inference_mode()
    with ctx:
        assert torch.is_inference_mode_enabled()


def test_override_returns_no_grad():
    ctx = NoGradPlatform.inference_mode()
    with ctx:
        t = torch.randn(2, 2)
        assert not t.requires_grad
        # no_grad does not enable inference mode
        assert not torch.is_inference_mode_enabled()


def test_default_inference_mode_as_decorator():
    @Platform.inference_mode()
    def fn():
        return torch.is_inference_mode_enabled()

    assert fn()


def test_override_no_grad_as_decorator():
    @NoGradPlatform.inference_mode()
    def fn():
        return torch.is_inference_mode_enabled()

    assert not fn()
