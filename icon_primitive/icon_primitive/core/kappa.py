"""
ICON-Primitive Core: κ (kappa) 정의 및 측정

κ = I(X; Z̃) / d_z

여기서:
- X: primitive 입력
- Z = f(X): primitive 출력
- Z̃ = Z + ε: 측정 채널 통과 후 출력
- ε ~ N(0, (σ · RMS(Z))² I)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

from icon_primitive.core.noise_channel import NoiseChannel
from icon_primitive.core.mi_estimators import get_estimator


@dataclass
class KappaConfig:
    """κ 측정 설정 (락)"""
    
    # 측정 채널
    sigma: float = 0.10
    sigma_mode: str = "rms_scaled"
    
    # 평가 샘플
    n_eval: int = 8192

    # MI estimator train/test split (protocol-locked)
    mi_train: int = 4096
    mi_test: int = 4096
    
    # MI estimator
    estimator_name: str = "infonce"
    estimator_config: Dict[str, Any] = field(default_factory=dict)
    
    # Seeds
    mi_seed: int = 0
    
    # Sanity checks
    run_sanity: bool = True
    permutation_mi_threshold: float = 0.1  # MI ≈ 0


@dataclass
class KappaResult:
    """κ 측정 결과"""
    
    # 핵심 결과
    kappa_raw: float           # I(X; Z̃)
    kappa_per_dim: float       # I(X; Z̃) / d_z
    d_z: int                   # 출력 차원
    
    # Estimator별 결과
    by_estimator: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 진단
    saturation_flag: bool = False
    saturation_margin: Optional[float] = None
    
    # Sanity check
    permuted_mi: Optional[float] = None
    sanity_passed: bool = True
    
    # 메타데이터
    n_eval: int = 0
    sigma: float = 0.0


def compute_rms(Z: torch.Tensor) -> float:
    """
    RMS(Z) = sqrt(mean(Z²))
    batch + dim 전체 평균
    """
    return torch.sqrt(torch.mean(Z ** 2)).item()


def compute_kappa(
    X: torch.Tensor,
    Z: torch.Tensor,
    config: KappaConfig,
    device: torch.device = torch.device("cpu"),
) -> KappaResult:
    """
    κ = I(X; Z̃) / d_z 계산
    
    Args:
        X: primitive 입력 [N, d_x]
        Z: primitive 출력 [N, d_z]
        config: KappaConfig
        device: 연산 장치
    
    Returns:
        KappaResult
    """
    assert X.shape[0] == Z.shape[0], "X와 Z의 샘플 수가 일치해야 함"
    assert X.shape[0] == config.n_eval, f"N_eval={config.n_eval} 고정 위반"
    assert config.mi_train + config.mi_test == config.n_eval, "mi_train+mi_test must equal n_eval"
    
    N = X.shape[0]
    d_z = Z.shape[1] if Z.dim() > 1 else 1
    
    # 1. 측정 채널 적용: Z̃ = Z + ε
    noise_channel = NoiseChannel(
        sigma=config.sigma,
        sigma_mode=config.sigma_mode,
        seed=config.mi_seed,
    )
    Z_tilde = noise_channel(Z)
    
    # 2. MI 추정 (protocol split: train critic on first mi_train, evaluate on remaining)
    torch.manual_seed(config.mi_seed)
    np.random.seed(config.mi_seed)

    estimator = get_estimator(name=config.estimator_name, config=config.estimator_config, device=device)

    X_train = X[: config.mi_train].to(device)
    Z_train = Z_tilde[: config.mi_train].to(device)
    X_test = X[config.mi_train :].to(device)
    Z_test = Z_tilde[config.mi_train :].to(device)

    estimator.fit(X_train, Z_train)
    mi_result = estimator.estimate(X_test, Z_test)
    
    kappa_raw = mi_result["mi"]
    kappa_per_dim = kappa_raw / d_z
    
    # 3. Saturation check (InfoNCE)
    saturation_flag = False
    saturation_margin = None
    
    if config.estimator_name == "infonce":
        log_batch = float(mi_result.get("log_batch", np.log(config.estimator_config.get("batch_size", 512))))
        margin = float(config.estimator_config.get("saturation_margin", 0.1))
        saturation_margin = log_batch - kappa_raw
        saturation_flag = saturation_margin < margin
    
    # 4. Sanity check: permuted Z
    permuted_mi = None
    sanity_passed = True
    
    if config.run_sanity:
        perm_idx = torch.randperm(config.mi_test)
        # permute within test split (keep train split untouched)
        Z_permuted = Z[config.mi_train :][perm_idx]
        Z_tilde_perm = noise_channel(Z_permuted)
        perm_result = estimator.estimate(X_test, Z_tilde_perm.to(device))
        permuted_mi = perm_result["mi"]
        
        if permuted_mi > config.permutation_mi_threshold:
            sanity_passed = False
    
    return KappaResult(
        kappa_raw=kappa_raw,
        kappa_per_dim=kappa_per_dim,
        d_z=d_z,
        by_estimator={
            config.estimator_name: {
                "raw": kappa_raw,
                "per_dim": kappa_per_dim,
            }
        },
        saturation_flag=saturation_flag,
        saturation_margin=saturation_margin,
        permuted_mi=permuted_mi,
        sanity_passed=sanity_passed,
        n_eval=N,
        sigma=config.sigma,
    )


def compute_constant_C(
    kappa_variant: float,
    kappa_base: float,
) -> float:
    """
    상수 C = κ_variant / κ_base
    """
    assert kappa_base > 0, "κ_base must be positive"
    return kappa_variant / kappa_base


class KappaMeasurer:
    """
    κ 측정 파이프라인
    
    모델의 특정 지점(tap)에서 X, Z를 추출하고 κ를 계산
    """
    
    def __init__(
        self,
        config: KappaConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.config = config
        self.device = device
    
    def measure(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        tap_input: str = "input",
        tap_output: str = "output",
    ) -> KappaResult:
        """
        모델에서 κ 측정
        
        Args:
            model: 측정 대상 모델
            dataloader: 평가 데이터 로더 (N_eval 샘플)
            tap_input: 입력 측정 지점
            tap_output: 출력 측정 지점
        
        Returns:
            KappaResult
        """
        model.eval()
        model.to(self.device)
        
        X_list = []
        Z_list = []
        
        # Hook을 사용하여 중간 출력 캡처
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # 필요한 경우 hook 등록
        # (실제 구현에서는 모델 구조에 맞게 수정 필요)
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                # Forward pass
                output = model(x)
                
                # 입력/출력 수집
                X_list.append(x.cpu())
                Z_list.append(output.cpu())
        
        X = torch.cat(X_list, dim=0)[:self.config.n_eval]
        Z = torch.cat(Z_list, dim=0)[:self.config.n_eval]
        
        # Flatten if needed
        if X.dim() > 2:
            X = X.view(X.shape[0], -1)
        if Z.dim() > 2:
            Z = Z.view(Z.shape[0], -1)
        
        return compute_kappa(X, Z, self.config, self.device)
