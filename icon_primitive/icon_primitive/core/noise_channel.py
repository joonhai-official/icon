"""
ICON-Primitive Core: 측정 채널 (Noise Channel)

Z̃ = Z + ε
ε ~ N(0, (σ · RMS(Z))² I)

MI 발산 공격을 차단하기 위해 측정 채널을 κ 정의에 포함

Note on Reproducibility:
    - Generator는 device별로 하나씩 생성되며, seed가 주어지면 deterministic합니다.
    - 같은 NoiseChannel 인스턴스에서 forward()를 여러 번 호출하면 연속된 noise가 생성됩니다.
    - 동일한 noise를 재현하려면 새 NoiseChannel 인스턴스를 생성하거나 reset_generator()를 호출하세요.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class NoiseChannel(nn.Module):
    """
    측정 채널: Z̃ = Z + ε
    
    ε ~ N(0, (σ · RMS(Z))² I)
    RMS(Z) = sqrt(mean(Z²))
    """
    
    def __init__(
        self,
        sigma: float = 0.10,
        sigma_mode: str = "rms_scaled",
        seed: Optional[int] = None,
    ):
        """
        Args:
            sigma: 노이즈 스케일 (기본 0.10)
            sigma_mode: 스케일링 방식 ("rms_scaled" | "fixed")
            seed: 난수 시드 (재현성)
        """
        super().__init__()
        
        assert sigma >= 0, "sigma must be non-negative"
        assert sigma_mode in ["rms_scaled", "fixed"], f"Unknown sigma_mode: {sigma_mode}"
        
        self.sigma = sigma
        self.sigma_mode = sigma_mode
        self.seed = seed
        
        # NOTE: torch.Generator is device-specific for CUDA.
        # We create generators lazily per device on first call.
        self._generators = {}  # device_str -> torch.Generator
    
    def compute_rms(self, Z: torch.Tensor) -> torch.Tensor:
        """
        RMS(Z) = sqrt(mean(Z²))
        batch + dim 전체 평균
        """
        return torch.sqrt(torch.mean(Z ** 2))
    
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z̃ = Z + ε
        
        Args:
            Z: primitive 출력 [N, d_z] or [N, C, H, W]
        
        Returns:
            Z̃: 노이즈가 추가된 출력
        """
        if self.sigma == 0:
            return Z
        
        # RMS 계산
        if self.sigma_mode == "rms_scaled":
            rms = self.compute_rms(Z)
            noise_std = self.sigma * rms
        else:  # fixed
            noise_std = self.sigma
        
        # 노이즈 생성 (deterministic if seed is provided)
        if self.seed is not None:
            dev_key = str(Z.device)
            gen = self._generators.get(dev_key)
            if gen is None:
                gen = torch.Generator(device=Z.device)
                gen.manual_seed(int(self.seed))
                self._generators[dev_key] = gen
            epsilon = torch.randn(Z.shape, device=Z.device, dtype=Z.dtype, generator=gen) * noise_std
        else:
            epsilon = torch.randn_like(Z) * noise_std
        
        return Z + epsilon
    
    def get_config(self) -> dict:
        """설정 반환 (receipt용)"""
        return {
            "type": "gaussian",
            "sigma": self.sigma,
            "sigma_mode": self.sigma_mode,
            "seed": self.seed,
        }

    def reset_generator(self) -> None:
        """Generator를 초기 상태로 리셋 (동일한 noise 시퀀스 재현용)"""
        self._generators = {}


def create_noise_channel(
    sigma: float = 0.10,
    sigma_mode: str = "rms_scaled",
    seed: Optional[int] = None,
) -> NoiseChannel:
    """NoiseChannel 팩토리 함수"""
    return NoiseChannel(sigma=sigma, sigma_mode=sigma_mode, seed=seed)


# Sanity sweep sigmas
SANITY_SIGMAS = [0.05, 0.10, 0.20]


def run_sigma_sweep(
    Z: torch.Tensor,
    X: torch.Tensor,
    estimator,
    sigmas: list = SANITY_SIGMAS,
    seed: int = 0,
) -> dict:
    """
    σ sweep으로 C의 안정성 검증
    
    Args:
        Z: primitive 출력
        X: primitive 입력
        estimator: MI estimator
        sigmas: 테스트할 σ 값들
        seed: 난수 시드
    
    Returns:
        sigma → MI 매핑
    """
    results = {}
    
    for sigma in sigmas:
        channel = NoiseChannel(sigma=sigma, sigma_mode="rms_scaled", seed=seed)
        Z_tilde = channel(Z)
        
        mi_result = estimator.estimate(X, Z_tilde)
        results[sigma] = {
            "mi": mi_result["mi"],
            "per_dim": mi_result["mi"] / Z.shape[-1],
        }
    
    return results
