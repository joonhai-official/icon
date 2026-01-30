# ICON‑Primitive 1F 조합(15개) 자동 생성 규칙 v1.1

이 문서는 **1F(독립성 검증)**을 “충돌 없이 / 재현 가능하게 / 통계적으로 의미 있게” 구성하기 위한 **규칙(=프리레지스터)**이다.

핵심은 두 가지다.
1) **곱셈 독립성**: `kappa_rel ≈ Π C_i`
2) **상호작용(교차항)**이 생기면 감지 가능해야 한다.

---

## A. 공통 원칙(절대 규칙)

1. **baseline 고정**
   - act=relu, norm=none, prec=fp32, skip=none (vector probe)
   - linear_type까지 포함하는 경우: linear_type=dense, budget=shape_matched (spatial probe)

2. **측정 채널 고정**
   - `Z_tilde = Z + N(0, (sigma*RMS(Z))^2 I)`
   - sigma는 1F 전체에서 동일(메인 sigma=0.10)

3. **residual 스케일링 고정**
   - `y=(x+f(x))/sqrt(2)` (variance_preserving)

4. **precision은 PTQ로만 정의(1F에서도 동일)**
   - 학습은 fp32로, 변환 후 kappa 측정

---

## B. 15개(core) 조합 생성 규칙

### 목표 커버리지
- activation: 6개 이상 (relu 포함)
- norm: 4개 이상 (none 포함)
- precision: 4개 이상 (fp32 포함)
- skip: none과 residual 모두 포함

### 구조(권장)
15개를 5개 블록으로 분해하면 설계가 안정적이다.

**Block 1 — Act×Norm (prec는 fp32 고정, skip none)**
- 목적: act/norm 상호작용 탐지
- 4개

**Block 2 — Prec stress (act/norm은 대표 1~2개로 고정, skip none)**
- 목적: precision이 다른 요소와 곱셈으로 잘 붙는지
- 4개

**Block 3 — Skip stress (residual만 켜서 확인)**
- 목적: skip이 act/norm/prec와 곱셈으로 붙는지
- 4개

**Block 4 — Nonlinearity edge cases**
- 목적: tanh/sigmoid/identity 같은 극단 분포에서 안정성
- 3개

합계 4+4+4+3=15.

---

## C. “충돌 없이” 자동 생성하기 위한 체크리스트

자동 생성기는 아래 제약을 만족해야 한다.

1) **동일 config 중복 금지**
- (act,norm,prec,skip) 4‑tuple이 중복되면 안 됨

2) **단일 요소만 바뀌는 페어를 최소 4개 확보**
- 예: (gelu,ln,fp32,none) vs (gelu,ln,fp16,none)
- 이렇게 해야 각 상수가 독립적으로 “식별”된다.

3) **residual이 들어간 config는 최소 4개**
- skip 상수의 곱셈성을 충분히 검사

4) **precision은 최소 3종 이상**
- fp16/bf16/int8/int4 중 최소 3종

5) **norm은 최소 3종 이상**
- ln/rms/bn/gn 중 최소 3종

---

## D. LinearType 독립성(확장) 생성 규칙

core 15개는 vector‑probe 기반이라 linear_type을 넣기 어렵다(공정성/shape가 흔들림).
그래서 아래 **8개 확장**으로 `C_type`의 곱셈성을 닫는다.

- baseline pair: (relu, none, fp32, none)
- stress pair: (gelu, ln, fp16, residual)
- linear_type: {dense, conv1x1, conv3x3, depthwise}

=> 2(pairs) × 4(types) = 8개

이 확장은 “심사자 공정성 공격”을 크게 줄여준다.

---

## E. 출력 포맷

자동 생성의 출력은 아래 2개로 고정한다.
- YAML: `ICON_Primitive_Independence_Configs_v1.1.yaml`
- CSV:  `ICON_Primitive_Independence_Configs_v1.1.csv`

둘 다 동일한 config ID를 유지해야 한다.
