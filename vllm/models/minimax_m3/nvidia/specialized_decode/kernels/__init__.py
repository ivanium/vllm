# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused attention front of vLLM #66324's specialized decode (its stage
`front`, `front.py` + `front_kernel.py`) with the peer transport, push exchange,
target and cache-format modules it needs, taken from that PR as merged
(b6dfb9543e). The low-batch decode runs it on its sparse layers
(`runtime/front.py`); #66324's stage registry and its other kernels are not
part of this package.
"""
