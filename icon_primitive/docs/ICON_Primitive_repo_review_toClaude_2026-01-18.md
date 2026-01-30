# ICON-Primitive v1.1 Repo Review — To Claude (2026-01-18)

대상: `icon_primitive_v1.1.zip` 스냅샷 기준.

## 0) TL;DR
- **설계/스펙 문서화는 매우 잘 됨**: `configs/base_protocol.yaml`에서 κ 정의(측정 채널), 학습/seed/estimator/skip 공정성까지 “락”이 거의 다 들어가 있음.
- **핵심 코어도 방향 맞음**: `NoiseChannel`(σ·RMS), `compute_kappa`, Vector/Spatial probe에서 frozen stem + (X,Z) 캡처.
- **하지만 실행 파이프라인 연결이 아직 부족**해서, 지금 상태로는 1편 전 범위(1A~1G)를 **스펙대로 재현성 게이트 포함해 돌릴 수 없음**.

---

## 1) 이미 잘 되어 있는 것 (KEEP)
### 1.1 κ 정의/측정 채널
- `icon_primitive/core/noise_channel.py`: `Z̃ = Z + ε`, `ε ~ N(0,(σ·RMS(Z))^2 I)` 구현 O
- `icon_primitive/core/kappa.py`: `κ = I(X;Z̃)/d_z` 구현 O, permuted-Z sanity 포함 O

### 1.2 Probe 공정성
- `icon_primitive/models/probes.py`
  - FrozenStem로 stem 파라미터 고정 O
  - VectorProbe에서 (X,Z) capture O
  - Residual skip에 variance_preserving(1/sqrt(2)) 옵션 O
  - SpatialProbe(LinearType 비교) 골격 O

### 1.3 Config 설계
- `configs/base_protocol.yaml`: 스펙 락 구조가 매우 좋음.
- `configs/sections/*`: 1A~1G 요구사항을 문서화/프리레지스터 형태로 잘 정리.

---

## 2) P0 — 반드시 보완해야 하는 핵심 빈틈(안 하면 1편 주장 흔들림)

### P0-1. Config→실행기 연결이 거의 없음
현재 `scripts/run_single.py`는 string 포함 여부로 (activation/normalization/skip 일부)만 처리하고,
- 1B(SpatialProbe), 1D(Precision/PTQ), 1F(Independence), 1G(Robustness) 실행이 사실상 미구현.
- 1C의 pre_norm/post_norm tap 측정 요구도 미구현.

**필수 조치**
- `run_single.py`를 “section config YAML”을 읽어서 **probe/model/training/measurement를 생성하는 엔진**으로 바꾸기.
- 섹션별 요구(탭/paired ratio/track 분리/ptq 변환)를 config 기반으로 분기.

### P0-2. Eval indices / calibration indices 고정 + hash 기록 미구현
- README/프로토콜에는 eval_indices 고정이 핵심인데, 코드에선 test loader에서 앞에서부터 N_eval만 취함.
- receipt에는 `eval_indices_hash`, `data_hash`가 `TBD`로 남아 있음.

**필수 조치**
- `assets/eval_indices/{dataset}_test_8192.npy` 생성 + 로드해서 sampler로 사용.
- PTQ calibration indices도 `assets/calib_indices/{dataset}_train_1024.npy`로 고정.
- receipt에 `eval_indices_hash`, `calibration_indices_hash`, `stem_hash`를 반드시 기록.

### P0-3. Receipt schema gate가 “진짜 schema 검증”이 아님
- `icon_primitive/utils/receipt.py`의 validate는 key 존재만 체크.
- repo는 `jsonschema` dependency가 있으나 실제 검증 미사용.

**필수 조치**
- 프로젝트에 `Receipt_Schema_v1.1.json`을 포함하고, `jsonschema.validate()`로 강제.
- 미통과 receipt는 aggregator에서 자동 제외.
- receipt에서 `TBD` 제거 (모든 hash/estimator config/σ/N_eval/seed 기록).

### P0-4. Warmup+cosine 스케줄 미적용 (프로토콜 위반)
- `base_protocol.yaml`은 warmup_cosine인데, `scripts/run_single.py`는 상수 lr로 100 epoch.

**필수 조치**
- warmup+cosine scheduler 구현 후 모든 run에 적용.

### P0-5. KSG 구현이 N_eval=8192에서 사실상 불가능(O(N^2))
- `icon_primitive/core/mi_estimators.py`의 KSG는 n_x/n_z를 브루트포스로 계산 → 8192에서 시간 폭발.

**필수 조치(권장 구현)**
- `sklearn.neighbors.KDTree(metric='chebyshev')` 사용해 `query_radius(..., r=eps)`로 **배열 radii**를 한번에 처리:
  - joint eps는 `kneighbors`로 구하고,
  - `n_x = KDTree(X).query_radius(X, r=eps-1e-15, count_only=True)-1`
  - `n_z = KDTree(Z).query_radius(Z, r=eps-1e-15, count_only=True)-1`

### P0-6. 1D Precision/PTQ 코드 없음
- configs에는 PTQ 스펙이 있는데, repo에는 quantization 모듈이 없음.

**필수 조치**
- fp16/bf16: cast 변환
- int8/int4: weight per-channel symmetric + activation per-tensor symmetric fake-quant (고정 calibration indices)
- receipt에 observer/clipping stats 기록

### P0-7. 1E dense_concat_projected 미지원
- `configs/sections/1E_skip.yaml`에 projected 버전이 있는데 VectorProbe는 현재 dense_concat만 지원.

**필수 조치**
- `skip == 'dense_concat_projected'` 지원(512→256 projection 포함)
- main table은 projected를 사용(공정성)

### P0-8. run_id 충돌 위험
- run_id가 날짜+section+variant+seed라 같은 날 재실행 시 overwrite 가능.

**필수 조치**
- run_id에 `config_hash[:8]` 또는 timestamp(초 단위) 포함.

---

## 3) P1 — 강추(넣으면 논문 방어력 급상승)
1) 1F log-space interaction 회귀(γ_ij) 자동 리포트.
2) 1G 실패 시 조건부 상수 split(Estimator>Width>Dataset) 자동화.
3) Aggregator를 constants 템플릿 구조로 출력(메타데이터+receipt 링크 포함).
4) 최소 단위 테스트 추가:
   - NoiseChannel RMS scaling
   - compute_kappa N_eval assert
   - permuted-Z MI≈0
   - receipt schema validation

---

## 4) Definition of Done (1편 “돌릴 수 있다” 기준)
- Base153(또는 확장 포함) manifest를 config 기반으로 완주
- 모든 run에서 receipt schema 통과
- receipt에 hashes(config/data/eval_indices/stem), σ/N_eval/estimator+hyper, seeds 기록
- 1A~1E 상수표 + CI 출력
- 1F: κ_pred vs κ_meas error 기준 자동 판정
- 1G: 12조건 std(C)<3% 자동 판정

---

## 5) 빠른 구현 우선순위 제안
1) eval_indices + receipt schema gate + run_id 충돌 해결
2) runner를 section-config 기반으로 통합
3) KSG 최적화(KDTree)
4) 1E projected concat + 1C taps
5) 1D PTQ 구현
6) aggregator 확장(1F/1G 판정 포함)

