# ICON‑Primitive (1편) v1.1 — 빈틈 봉쇄 “마스터 체크리스트” + Claude 전달용 실행 지시서

이 문서는 **ICON 시리즈 1편(Primitive 상수)**을 *공정성/재현성 공격 불가 수준*으로 잠그기 위해, 1편에서 **반드시 보완/고정해야 할 남은 요소**를 빠짐없이 정리한 **프리레지스터(Pre‑register) + 구현 체크리스트**입니다.

> 핵심 철학: 1편은 “절대 κ”가 아니라 **ratio 상수 C**가 주장이므로, 비교 가능성(공정성)과 재현성(프로토콜 고정)을 최우선으로 설계한다.

---

## 0) 1편의 불변 목표
1편은 다음을 성립시키는 것이 목적이다.

1) Primitive 요소(Activation / LinearType / Norm / Precision / Skip)의 정보용량을 **비율 상수 C**로 정의할 수 있다.
2) 상수들이 **곱셈 조합(독립성)**으로 합성되어 전체 κ를 예측한다.
3) 조건 변화(데이터셋/폭(width)/MI estimator 등)에도 **비율 C가 안정적(robust)**이다.

---

## 1) P0(필수) — 이게 없으면 1편은 공격받아 무너짐

### P0‑1. κ 정의를 “측정 채널 포함”으로 고정 (MI 발산 공격 차단)
연속값 결정론 매핑의 MI 발산(무한대) 공격을 차단하기 위해 **측정 채널을 κ 정의에 포함**한다.

- 입력: X (primitive 입력)
- 출력: Z = f(X)
- 측정 채널: **Z̃ = Z + ε**
  - ε ~ N(0, (σ·RMS(Z))^2 I)
  - RMS(Z) = sqrt(mean(Z^2))
- κ = I(X; Z̃) / d_z
- C_variant = κ_variant / κ_base

고정값:
- 메인 σ = 0.10
- sanity σ sweep = {0.05, 0.20}

> sanity sweep의 목적은 “값 맞추기”가 아니라 **C의 순위/대략 비율이 유지되는지** 확인하는 것.

---

### P0‑2. MI 추정기(InfoNCE/MINE/KSG) 운영 규칙을 “프로토콜로 잠금”
**추정기 이름만 적으면 재현성/공정성이 성립하지 않는다.**
각 추정기별로 *구조/하이퍼/학습 스텝/seed/진단*을 프리레지스터로 고정한다.

#### (a) InfoNCE (Primary)
- eval pairs: N_eval=8192 고정(인덱스 고정)
- negatives 구성 방식 고정(같은 batch 내 in‑batch negatives 등)
- critic 네트워크 구조/학습 step/lr/seed 고정
- **포화(saturation) 진단 필수**:
  - MI_est < log(B_neg) − margin (예: 0.1) 를 만족해야 ‘정상’
  - 포화면 `saturation_flag=true` 기록
  - 포화 run은 robustness 통계에서 별도 표기(또는 사전등록 규칙에 따라 제외)

#### (b) MINE (Secondary)
- critic 구조/step/lr/EMA 고정
- early stopping 금지(비교 가능성 깨짐)
- 내부 seed 고정(mi_seed)

#### (c) KSG (Tertiary)
- 고차원 취약성 봉쇄: **고정 랜덤 직교 투영(시드 고정)으로 32‑d 축소 후 KSG**
- k=5 고정

#### (d) Sanity tests (필수)
- (X, permuted Z)에서 MI ≈ 0
- random label / random network에서 C가 상수처럼 나오지 않는지
- estimator 간 C의 ‘방향/순위’ 최소 일치(완전 일치는 요구하지 않되, 정반대면 경고)

---

### P0‑3. “primitive만 바뀌게” 만드는 공정한 입력 분포: Frozen Stem(프로브) 고정
Primitive 상수 비교에서 가장 흔한 공격은 “입력 분포/표현 분포가 run마다 달라져서 생긴 차이”다.
그래서 **primitive 입력 X가 항상 같은 분포/차원**이 되도록 **Frozen Stem**을 고정한다.

#### Vector‑probe (1A/1C/1D/1E/1F/1G 메인)
- E0(고정, frozen): raw input → X ∈ R^{256}
- 테스트 블록 Bθ: primitive 포함(여기만 변경)
- head Hθ: 분류/회귀 헤드(학습 가능)

고정 규칙:
- E0는 seed 고정 + 완전 frozen
- primitive 입력 X는 항상 256‑d
- κ 측정은 primitive 출력 Z(섹션별로 “측정 지점”을 명시)

#### Spatial‑probe (1B LinearType 전용)
- S0(고정, frozen): raw image → X ∈ R^{16×8×8}
- 테스트 op: Dense/Conv1×1/Conv3×3/Depthwise
- head: GAP + Linear

고정 규칙:
- S0 seed 고정 + 완전 frozen
- 입력 H×W×C 및 출력 shape 정의를 config로 잠금

---

### P0‑4. 학습 스펙(optimizer/schedule/epochs/batch) 완전 고정
“GELU가 더 잘 학습돼서 κ가 높아진 것” 같은 공격을 막기 위해 학습 스펙은 **전 실험 공통으로 고정**한다.

- optimizer: AdamW
- lr/schedule: (예: warmup 5 epochs + cosine) — **하나로 고정**
- epochs: 고정(early stopping OFF)
- batch_size, weight_decay, grad_clip: 고정
- training dtype: FP32 고정(1D precision은 *학습 FP32, 변환 후 측정*)

추가 고정:
- data augmentation은 1편에서 원칙적으로 OFF(필요하면 전 조건에 동일 적용)

---

### P0‑5. 초기화(init) 정책 고정
Activation마다 유리한 init을 넣으면 공정성 공격 포인트가 된다.
따라서 init은 **중립적인 단일 규칙**으로 고정한다.

권장 예:
- Linear/Conv: orthogonal init + scale to unit variance pre‑act
- bias: 0

---

### P0‑6. 평가 샘플 인덱스(eval_indices) 고정
- N_eval=8192 고정
- 어떤 데이터 포인트로 κ를 계산하는지 **인덱스 고정**
- receipt에 `eval_indices_hash` 기록

---

### P0‑7. Seed 3종 분리(모델/데이터/MI)
- model_seed: 3개(0/1/2)
- data_seed: 고정 1개(+ robustness에서 추가 가능)
- mi_seed: 고정 1개(MI estimator 내부 학습/샘플링)

---

### P0‑8. 1E Skip의 “가짜 상승” 차단
- Residual: y = (x + f(x)) / sqrt(2) **(variance‑preserving 스케일 고정)**
- Concat: 차원 증가 효과 분리
  - concat 후 **projection으로 d_z를 baseline으로 복원하는 버전**을 같이 측정

---

### P0‑9. 1D Precision(PTQ) 재현성 잠금
Precision 상수의 정의를 **PTQ**로 못박는다.

- FP32로 학습 → 변환(예: FP16/INT8/INT4) → κ 측정

Quant 세부 스펙(반드시 고정):
- weights: symmetric, per‑channel (또는 per‑tensor) — 하나로 고정
- activations: symmetric, per‑tensor — 고정
- calibration: 샘플 수/인덱스/seed 고정
- clipping/observer 방식 고정(예: percentile, minmax 등)

---

### P0‑10. 1B LinearType 공정성: “2트랙 분리” 없으면 공격 100%
LinearType은 가장 공격받기 쉬운 구간이라 **2트랙 분리**를 필수로 한다.

- Track‑A: shape‑matched(공간 입력 유지) — inductive bias 포함
- Track‑B: budget‑matched(params 또는 FLOPs 매칭) — 공정성 방어

보고 원칙:
- 두 결과를 **같은 표에 섞지 말고** 별도 상수로 보고
  - C_linear_type_shape
  - C_linear_type_budget

---

### P0‑11. Receipt(영수증) 기반 게이트 = “실험 인정 조건”
- 모든 run은 receipt.json을 남겨야 한다.
- receipt가 schema를 통과하지 못하면 **그 run은 집계에서 자동 제외**.

receipt 필수 항목:
- git_commit / config_hash / data_hash / eval_indices_hash / stem_hash
- primitive 옵션 + 섹션(1A~1G)
- σ, N_eval, estimator type + hparams + saturation flag
- seeds(model/data/mi)
- environment(pytorch/cuda/gpu)

---

### P0‑12. 집계/통계 정의 고정 (숫자가 바뀌는 함정 제거)
- seed 집계 방식 고정(예: `mean(C_seed)`; ratio of means 금지 등)
- bootstrap 고정:
  - resample 단위(예: eval pairs)
  - 반복 횟수(예: 2000)
  - CI 방식(예: percentile 2.5/97.5)
- outlier 제거 금지(또는 사전등록된 규칙만)

---

## 2) P1(강추) — 넣으면 논문 방어력 급상승

### P1‑1. 1F 독립성의 “통계 버전”(log‑space interaction test)
- 메인 판정: κ_pred vs κ_meas 오차 기준
  - 평균<5%, 최대<15%, 개별<10%
- 추가(부록): log‑space에서 상호작용항(γ_ij) 검정
  - log κ = α + Σ log C_i + Σ γ_ij x_i x_j + ε

> 1F가 완전 독립이 아니어도 **준독립(보정항)**으로 자연스럽게 전환할 출구를 미리 만든다.

---

### P1‑2. 1G Robustness를 “정확한 조건표”로 프리레지스터
문서에 ‘12조건’만 적으면 선택 편향 공격을 받는다. 조건을 **명시적 매트릭스**로 고정한다.

권장 12조건(예시, 고정):
- datasets: {MNIST, CIFAR‑10, ImageNet‑subset}
- widths: {256, 1024}
- MI estimator: {InfoNCE, KSG}
→ 3×2×2 = 12

그리고 robustness 대상은 최소 1개 sentinel(예: GELU)로 시작하고,
방어력을 더 올리려면 카테고리별로 1개씩 추가(sentinel set):
- norm sentinel: LayerNorm
- precision sentinel: FP16
- skip sentinel: Residual
(단, run 폭발 방지를 위해 축을 줄여서 수행)

---

### P1‑3. 실패 시 전환 규칙을 자동화(Plan‑B를 설계로 흡수)
- 1G 실패 시: 조건부 상수로 최소 분할(데이터/폭/추정기 중 최소 축) 규칙 고정
- 1F 실패 시: interaction 항 추가 우선순위 규칙 고정

---

### P1‑4. 디렉토리/네이밍 컨벤션 고정(병렬 실행 안정화)
- run_id 규칙, 폴더 구조, 파일명 규칙
- 중복/누락/merge 사고 방지

---

## 3) P2(옵션) — 있으면 좋지만 1편 필수는 아님
- QAT(Quantization‑Aware Training) 상수(PTQ와 별도 표)
- data_seed robustness 추가 축
- 더 많은 width/dataset 확장

---

## 4) 섹션별 “측정 지점/비교 단위” 잠금 (자잘하지만 자주 공격받음)

### 4‑1. 1A Activation
- baseline: ReLU
- 측정 지점: activation 출력(Z)
- activation list를 config로 고정

### 4‑2. 1C Norm
- 측정 지점 2개(권장):
  - Z_pre: norm 전
  - Z_post: norm+act 후

### 4‑3. 1D Precision
- 학습: FP32 고정
- 변환 후 같은 입력 X로 κ 측정

### 4‑4. 1E Skip
- residual scaling 고정
- concat projection 버전 포함

### 4‑5. 1B LinearType
- Track‑A/Track‑B 분리 고정
- 입력/출력 shape을 config에 명시

---

## 5) 1편에서 “구현/패키징”으로 남은 실물 산출물(Deliverables)

### D‑1. Config Pack (필수)
- 1A Activation sweep configs
- 1B LinearType configs (Track‑A/Track‑B)
- 1C Norm sweep configs
- 1D Precision sweep configs (PTQ + calibration 고정)
- 1E Skip sweep configs (residual scaling + concat projection)
- 1F Independence configs (15 core + optional linear_type 8)
- 1G Robustness configs (12‑condition matrix + sentinel optional)

### D‑2. Runner / Orchestrator (필수)
- 단일 run 실행기: config → train → κ measure → receipt.json 저장
- 매니페스트 실행기: N configs × seeds 병렬 실행
- 실패 재시도 정책/중단 정책 고정

### D‑3. Collector / Aggregator (필수)
- receipt 폴더를 읽어 constants.yaml 생성
- bootstrap CI 계산
- 1F 판정(오차표 + 통과/실패)
- 1G 판정(std(C)<3% 등)

### D‑4. Report Generator (필수)
- 표/그림 자동 생성
  - 상수표(C + CI)
  - 1F scatter(κ_pred vs κ_meas) + error histogram
  - 1G 조건별 C 분산 시각화

### D‑5. QA 체크(필수)
- sanity tests 통과 여부
- InfoNCE saturation flag 통계
- estimator 간 방향 불일치 발견 시 경고

---

## 6) Claude에게 줄 “구체 실행 지시(복붙 프롬프트)”

아래 블록을 Claude에 그대로 붙여넣고, **코드/파일 산출물**까지 생성하도록 시키면 됩니다.

---

### ✅ Claude Prompt (copy/paste)

너는 ICON 시리즈 1편(ICON‑Primitive) 구현/실험 파이프라인 담당이다. 목표는 Primitive 상수 C를 공정성/재현성 공격 불가 수준으로 산출하는 것이다.

[필수 스펙]
1) κ 정의는 측정 채널 포함으로 고정한다:
- Z = f(X)
- Z̃ = Z + ε, ε ~ N(0, (σ·RMS(Z))^2 I)
- κ = I(X; Z̃)/d_z
- C = κ_variant/κ_base
- 메인 σ=0.10, sanity σ∈{0.05,0.20}

2) “primitive만 바뀌는 비교”를 위해 frozen stem을 고정한다:
- vector_probe: E0(frozen) → X∈R^256 → primitive block Bθ(변경) → head
- spatial_probe(1B): S0(frozen) → X∈R^{16×8×8} → linear_type op(변경) → head
- stem은 seed 고정 + 완전 frozen

3) 학습 스펙은 전 실험 공통 고정:
- optimizer/lr schedule/epochs/batch/wd/grad_clip 고정
- early stopping OFF
- training dtype FP32 고정(precision 실험도 학습은 FP32)

4) 평가 샘플 인덱스(eval_indices)와 N_eval=8192를 고정하고 hash를 receipt에 기록한다.

5) seed는 model/data/mi로 분리하고 model_seed는 3개(0/1/2)로 돌린다.

6) MI estimator는 InfoNCE(Primary), MINE(Secondary), KSG(Tertiary) 3개를 지원하되 운영 규칙을 고정한다:
- InfoNCE: critic 구조/step/lr/seed 고정 + saturation 진단(MI_est < log(B)-0.1) 플래그 기록
- MINE: critic 구조/step/lr/EMA 고정, early stop 금지, mi_seed 고정
- KSG: 고정 랜덤 직교 투영으로 32‑d 축소 후 k=5 고정
- Sanity: (X, permuted Z)에서 MI≈0

7) 1E Skip 공정성:
- residual은 y=(x+f(x))/sqrt(2) 고정
- concat은 projection으로 d_z를 baseline으로 복원하는 버전을 같이 측정

8) 1D Precision은 PTQ로 고정:
- FP32 학습 → 변환(FP16/INT8/INT4 등) → κ측정
- quant calibration 샘플 인덱스/seed/방법 및 weight/act quant 스킴을 고정하고 receipt에 기록

9) 1B LinearType은 2트랙으로 분리:
- Track‑A shape‑matched
- Track‑B budget‑matched(params 또는 FLOPs 매칭)
- 결과를 같은 표에 섞지 말고 별도 상수로 보고

10) 모든 run은 receipt.json을 남기고 schema 검증을 통과해야 한다(미통과 run은 집계에서 자동 제외).
receipt에는 git_commit, config_hash, data_hash, eval_indices_hash, stem_hash, environment, σ/N_eval/estimator+hparams+saturation, seeds를 반드시 포함.

11) 집계/통계 규칙을 고정한다:
- seed 집계는 mean(C_seed) 방식
- bootstrap(예: 2000회, eval pairs resample, percentile 95% CI)
- outlier 제거 금지(또는 사전등록된 규칙만)

[산출물]
A) config pack 생성:
- 1A~1E sweep configs
- 1B Track‑A/Track‑B configs
- 1F independence configs(15 core + optional 8 linear_type)
- 1G robustness 12‑condition matrix configs

B) runner/orchestrator:
- config → 학습 → κ측정 → receipt 저장
- 매니페스트 병렬 실행

C) aggregator/report:
- receipt 폴더 → constants.yaml 생성
- 1F 판정(평균/최대/개별 오차)
- 1G 판정(std(C)<3%)
- 결과 표/그림 자동 생성

D) 디렉토리/네이밍 컨벤션 문서화 + 실행 예시 커맨드 제시.

[주의]
- outlier 제거 금지(또는 사전등록 규칙만)
- seed/indices/estimator hyper/hashes를 빠뜨리면 실험 무효

위 스펙을 만족하는 파일 구조와 코드/설정 파일을 생성하고, 실행 예시 커맨드까지 제시해라.

