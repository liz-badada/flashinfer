"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class LatentMoeProjectionConfig:
    """Prepared expert-projection tile and weight-cache policy."""

    bn: int = 32
    warps: int = 1
    stages: int = 3
    unroll: int = 1
    cache: str = "normal"
    hot_panels: int | None = None

    def __post_init__(self):
        if any(
            type(v) is not int for v in (self.bn, self.warps, self.stages, self.unroll)
        ):
            raise ValueError("integer launch parameters required")
        if self.hot_panels is not None and type(self.hot_panels) is not int:
            raise ValueError("integer hot-panel count required")
        if self.bn not in (16, 32, 64, 128) or self.warps not in (1, 2, 4, 8):
            raise ValueError("unsupported projection tile or warp count")
        if self.stages < 1 or self.unroll < 1:
            raise ValueError("stages and unroll must be positive")
        if self.cache not in ("normal", "first", "last"):
            raise ValueError("cache must be normal, first or last")
        if self.hot_panels is not None and (
            self.cache != "first" or self.hot_panels < 1
        ):
            raise ValueError("hot panels require first as the remaining-panel policy")


@dataclass(frozen=True)
class LatentMoeTP4Config:
    """Explicit launch, activation, overlap and communication capacities."""

    first: LatentMoeProjectionConfig = field(default_factory=LatentMoeProjectionConfig)
    second: LatentMoeProjectionConfig = field(default_factory=LatentMoeProjectionConfig)
    beta: float = 10.0
    up_beta: float = 10.0
    first_backend: str = "triton"
    down_mode: str = "split"
    down_bn: int = 128
    down_warps: int = 4
    down_k_group: int = 1
    up_mode: str = "split"
    up_k_group: int = 1
    shared_bn: int = 16
    mix_bn: int = 16
    pdl: bool = False
    bridge: bool = False
    overlap_shared: bool = True
    comm_tokens: int | None = None
    comm_hidden: int | None = None

    def __post_init__(self):
        integers = (
            self.down_bn,
            self.down_warps,
            self.down_k_group,
            self.up_k_group,
            self.shared_bn,
            self.mix_bn,
        )
        if any(type(v) is not int for v in integers):
            raise ValueError("integer launch parameters required")
        if any(
            v is not None and type(v) is not int
            for v in (self.comm_tokens, self.comm_hidden)
        ):
            raise ValueError("integer communication capacities required")
        if any(
            type(v) is not bool for v in (self.pdl, self.bridge, self.overlap_shared)
        ):
            raise ValueError("boolean execution flags required")
        if (
            not math.isfinite(self.beta)
            or self.beta <= 0
            or not math.isfinite(self.up_beta)
        ):
            raise ValueError("finite activation parameters and positive beta required")
        if self.first_backend not in ("triton", "rs") or self.down_mode not in (
            "split",
            "peer",
        ):
            raise ValueError("unsupported first projection or down mode")
        if self.up_mode not in ("split", "separate"):
            raise ValueError("unsupported up mode")
        if (
            self.down_bn not in (16, 128)
            or self.down_warps not in (1, 4)
            or any(g not in (1, 4) for g in (self.down_k_group, self.up_k_group))
        ):
            raise ValueError("unsupported raw projection tile or K group")
        if self.shared_bn not in (16, 128) or self.mix_bn not in (16, 128):
            raise ValueError("unsupported dense tile")
        if self.bridge and (not self.pdl or self.up_mode != "split"):
            raise ValueError("bridge requires PDL with split up projection")
        if any(v is not None and v < 1 for v in (self.comm_tokens, self.comm_hidden)):
            raise ValueError("communication capacities must be positive")


def _check_dimensions(
    tokens, hidden, latent, experts, width, shared_width, topk, config
):
    dimensions = (tokens, hidden, latent, experts, width, shared_width, topk)
    if any(type(v) is not int for v in dimensions) or min(dimensions) < 1:
        raise ValueError("positive dimensions required")
    if any(x % 128 for x in (hidden, latent, width, shared_width)):
        raise ValueError("matrix dimensions must preserve block128 boundaries")
    if hidden // 128 < topk or tokens * topk > 2**31 - 1:
        raise ValueError("fused routing requires H/128 >= topk and int32 slots")
    if topk > experts or experts * 2 * width > 2**31 - 1:
        raise ValueError("invalid expert routing or TMA row range")
    if config.down_mode == "peer" and latent % 512:
        raise ValueError("peer down requires four block128 row shards")
    if (latent // (4 if config.down_mode == "peer" else 1)) % config.down_bn:
        raise ValueError("down tile must divide local output rows")
    if config.first_backend == "rs" and config.first.hot_panels is not None:
        raise ValueError("native projection requires a uniform cache policy")
    for p, n in ((config.first, 2 * width), (config.second, latent)):
        if n % p.bn or (p.hot_panels is not None and p.hot_panels > tokens * topk):
            raise ValueError("projection tile or panel subset is out of range")
    if config.comm_tokens is not None and config.comm_tokens < tokens:
        raise ValueError("communication token capacity is too small")
    if config.comm_hidden is not None and config.comm_hidden < max(
        hidden, latent if config.down_mode == "peer" else hidden
    ):
        raise ValueError("communication width capacity is too small")
