# ICON-Primitive (1편) 실험 설계 스펙 v1.1

> 목적: “기초 연산(primitive)의 정보 용량”을 **상대 비율 상수(C)**로 정의하고, **공정성(fairness)**과 **재현성(reproducibility)**이 공격받지 않도록 “측정 채널/학습/평가/로그”를 완전히 고정(freeze)한다.

---

## 0. 용어/정의 (락 걸린 버전)

### 0.1 κ (information capacity per dim)
- 입력 표현: `X`
- 테스트 대상 연산(또는 블록) 출력: `Z = f(X)`
- **측정 채널(Measurement channel)**을 거친 출력: `Z̃`

**채널(고정)**
- `Z̃ = Z + ε`
- `ε ~ N(0, (σ · RMS(Z))^2 I)`
- `RMS(Z) = sqrt(mean(Z^2))` (batch+dim 전체 평균)

**κ 정의(고정)**
- `κ = I(X; Z̃) / d_z`

### 0.2 상수 C
- 같은 조건(데이터/width/estimator/σ/학습스케줄)에서
- `C_variant = κ_variant / κ_base`

### 0.3 베이스라인(락)
- Activation: `Linear + ReLU`
- Norm: `NoNorm (None)`
- Precision: `FP32`
- Skip: `None`
- Linear Type: `Dense(fully connected)`

---

## 1. Non‑negotiables (공정성/재현성 “절대 규칙”)

### 1.1 절대 고정(Frozen)
- 데이터 split, 전처리(mean/std), 샘플 인덱스, 셔플 seed
- 학습 스케줄(optimizer/lr/schedule/epochs/batch/clip/wd)
- 초기화 방식
- κ 측정 프로토콜(σ, N_eval, estimator, estimator 하이퍼, estimator seed)
- 코드 버전(git commit), 라이브러리 버전, 하드웨어 정보

### 1.2 seed 정책
- `model_seed` 3개 (예: 0,1,2)
- `data_seed` 1개(고정) + 필요시 2개 추가(robustness)
- `mi_seed` 1개(고정) (추정기 학습/샘플링의 변동 제거)

### 1.3 결정론 옵션(가능한 범위)
- `torch.backends.cudnn.deterministic=True`, `benchmark=False`
- 가능하면 `torch.use_deterministic_algorithms(True)`

---

## 2. 공통 프로토콜 (모든 섹션 공통)

### 2.1 데이터셋/전처리
- 기본 실험: CIFAR‑10
- 교차검증: MNIST / CIFAR‑10 / ImageNet‑subset

전처리(락)
- 입력 픽셀: `[0,1]` 스케일
- 데이터셋별 channel mean/std로 normalize
- augmentation: 기본은 OFF (교차검증에서만 별도 축으로 ON 가능)

### 2.2 “Primitive만 바뀌는” 프로브 네트워크
- 목표: 바꾼 건 오직 primitive 하나(또는 지정된 조합) 뿐이게 만들기

#### (A) Vector‑probe (1A/1C/1D/1E/1F/1G 대부분)
- **고정 입력 어댑터 E0**: raw input → 256‑d 벡터 `X`
  - MNIST: flatten(784) → Linear(784→256) (frozen, seed=123)
  - CIFAR: flatten(3072) → Linear(3072→256) (frozen, seed=123)
  - ImageNet‑subset: resize 64×64 후 flatten(12288) → Linear(12288→256) (frozen, seed=123)
- 테스트 블록 `Bθ`: (변형되는 primitive 포함)
- 분류 헤드 `Hθ`: Linear(d_z→#classes)

학습 파라미터: `Bθ`와 `Hθ`만 학습. `E0`는 완전 고정.

#### (B) Spatial‑probe (1B, 그리고 1G에서 linear type 검증 시)
- **고정 stem S0**: raw input → feature map `X ∈ R^{C×H×W}`
  - 예: CIFAR: Conv(3→16,k=3,s=2) + ReLU + Conv(16→16,k=3,s=2) + ReLU (frozen)
  - 출력: C=16, H=W=8
- 테스트 연산: Dense / Conv1×1 / Conv3×3 / Depthwise
- head: GAP + Linear(C→#classes)

### 2.3 학습(락)
- optimizer: AdamW
- lr: 3e‑4
- schedule: warmup 5epoch + cosine
- batch: 256
- epochs: 100
- weight_decay: 0.01
- grad_clip: 1.0
- loss: cross‑entropy
- training dtype: FP32 고정 (1D 제외)

### 2.4 초기화(락, activation 불리함 제거)
- Linear/Conv weight: orthogonal init + scale so that pre‑act variance≈1
- bias: 0

### 2.5 κ 측정(락)
- 측정 대상: `X = input to primitive`, `Z = output of primitive (정확히 지정된 지점)`
- 측정 데이터: test set에서 고정 인덱스 N_eval=8192
- noise channel: σ=0.10 (추가 sanity로 0.05/0.20)
- estimator: Primary=InfoNCE, Secondary=MINE, Tertiary=KSG(저차원 투영)

추정기 공통 규칙
- estimator training split: eval pairs 8192 → (train 4096 / test 4096)
- critic architecture/steps/lr 고정
- mi_seed 고정

InfoNCE(권장 스펙)
- critic: bilinear 또는 2‑layer MLP(256→512→256)
- steps: 2000
- batch: 512
- temperature τ=0.1
- saturation 체크: MI_est < log(batch) − 0.1

KSG(고정)
- X,Z 각각 fixed random orthogonal projection으로 32‑d로 축소(같은 seed)
- k=5

Sanity
- (X, permuted Z)에서 MI≈0 확인

---

## 3. 섹션별 설계

### 3.1 1A Activation constants
변수: act ∈ {ReLU, GELU, SiLU, Tanh, Sigmoid, Mish, Identity}
고정: Linear(256→256), NoNorm, FP32, NoSkip

측정지점: act output

성공 기준
- seed‑std(C) < 2%
- bootstrap 95% CI 폭 < 3%

추가(방어)
- σ sweep에서 C의 순위/대략 값이 유지

### 3.2 1B Linear type constants
변수: op ∈ {Dense, Conv1×1, Conv3×3, Depthwise}
고정: activation=ReLU, norm=None, fp32, no skip

Track‑A (shape‑matched)
- C_in=C_out=16, H=W=8
- Dense: flatten 1024→1024 (reshape)
- Conv1×1: 16→16
- Conv3×3: 16→16
- Depthwise: groups=16, 16→16

Track‑B (budget‑matched, 방어용)
- dense/conv의 파라미터 또는 FLOPs를 매칭하는 추가 설정 1개 이상

성공 기준
- seed‑std(C) < 2~3%

### 3.3 1C Normalization constants
변수: norm ∈ {None, LayerNorm, RMSNorm, BatchNorm, GroupNorm}
고정: activation=ReLU, fp32

측정지점 2개(권장)
- Z_pre = linear output
- Z_post = norm+act output

보고
- C_norm(pre/post) 둘 다 제공

### 3.4 1D Precision constants (PTQ로 고정)
변수: prec ∈ {FP32, FP16, BF16, INT8, INT4}
절차
1) FP32로 학습
2) precision 변환/양자화(고정 방식)
3) κ 측정

양자화(락)
- weight: symmetric per‑channel
- activation: symmetric per‑tensor
- calibration: training set 고정 1024 samples
- INT4는 간단한 uniform fake‑quant + clipping percentile(99.9)로 고정

### 3.5 1E Skip constants
변수: skip ∈ {None, Residual, Dense(concat)}
고정: Linear+ReLU, no norm, fp32

Residual (락)
- y = (x + f(x)) / sqrt(2)  (분산 보존)

Concat (락)
- concat 후 projection으로 d_z=256로 복원한 버전도 같이 측정
  - “차원 증가 효과”와 “skip 구조 효과” 분리

### 3.6 1F Independence verification
- 15개 조합 × 3 seeds
- κ_pred = κ_base × Π C_component

판정(메인)
- 평균 오차 < 5%
- 최대 오차 < 15%
- 개별 오차 모두 < 10%

판정(부록, 강추)
- log‑space 회귀로 상호작용항(γ_ij) 유의성 검사

### 3.7 1G Cross‑validation (robustness)
축
- dataset: MNIST/CIFAR/ImageNet‑subset
- estimator: KSG/MINE/InfoNCE
- width: 256/512/1024

운영(최소)
- 대표 상수 1개(예: GELU)로 12조건 × 3 seeds

운영(확장/방어)
- 각 카테고리에서 sentinel 1개씩(Activation/Norm/Precision/Skip/LinearType) 추가
- 단, 조건 수를 줄여(예: dataset×width만) 총 run을 통제

성공 기준
- std(C) < 3%
- 극단 조건에서도 C 유지(순위/방향성)

---

## 4. 분석/리포팅 규칙

### 4.1 집계
- seed별 C를 구한 뒤 평균/표준편차
- bootstrap(샘플 재추출)로 95% CI

### 4.2 필수 플롯
- 각 상수: C mean ± CI
- 1F: κ_measured vs κ_pred scatter + 상대오차 히스토그램
- 1G: 조건별 C boxplot

### 4.3 실패 시 규칙
- 상수 불안정 → “조건부 상수(conditional constants)”로 분기
- 원인 분해: σ, estimator, width, 데이터 스케일, 학습 안정성

---

## 5. Reproducibility 패키지(영수증)

### 5.1 run receipt(JSON) 필수
- run_id, git_commit, config_hash
- section(1A..1G), dataset, width
- model details(primitive 옵션)
- seeds(model/data/mi)
- training hyperparams
- κ measurement spec(σ, N_eval, estimator, critic spec)
- outputs(κ_base, κ_variant, C, CI, sanity flags)

### 5.2 아티팩트
- checkpoints(선택)
- eval pairs(X,Z) 요약 통계(RMS/mean/std)
- raw MI logs(critic loss curve)

---

## 6. 최종 산출물
- `ICON_Primitive_Constants_v1.1.yaml`
- `independence_report_v1.1.md`
- `robustness_report_v1.1.md`
- 재현성 문서: “동일 커밋/동일 seed로 ±ε 범위 내 재현”

