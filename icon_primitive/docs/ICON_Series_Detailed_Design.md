# ICON 시리즈 4편 상세 설계

---

# 1편: ICON-Primitive

## "기초 연산의 정보 용량 상수"

---

### 핵심 주장

> 단일 연산의 정보 용량은 **상대 비율(상수)**로 표현되며, 
> 이 상수는 조건(데이터, 스케일, 측정 방법)에 무관하게 안정적이다.

---

### 1A. Activation 상수

**목표:** 활성화 함수별 정보 보존 능력의 상대 비율 측정

**실험 구조:**
```
기준: Linear + ReLU = 1.000

측정 대상:
- Linear + GELU → C_gelu
- Linear + SiLU → C_silu  
- Linear + Tanh → C_tanh
- Linear + Sigmoid → C_sigmoid
- Linear + Mish → C_mish
- Linear + None (Identity) → C_none
```

**실험 설계:**
```python
# 네트워크 구조
class SingleLayerNet:
    def __init__(self, d_in, d_out, activation):
        self.linear = Linear(d_in, d_out)
        self.act = activation
    
    def forward(self, x):
        return self.act(self.linear(x))

# 측정
for act in [ReLU, GELU, SiLU, Tanh, Sigmoid, Mish, Identity]:
    net = SingleLayerNet(d_in=256, d_out=256, activation=act)
    train(net, data, epochs=100)
    κ = compute_kappa(net, test_data)  # I(X;Z) / d_out
    
# 상수 계산
C_act = κ_act / κ_relu
```

**실험량:** 7종 × 3 seeds = 21 runs

**검증 기준:**
- 3 seeds 간 표준편차 < 2%
- Bootstrap 95% CI 폭 < 3%

---

### 1B. Linear Type 상수

**목표:** 선형 변환 유형별 정보 용량 비율

**실험 구조:**
```
기준: Dense (fully connected) = 1.000

측정 대상:
- Conv 1×1 → C_conv1
- Conv 3×3 → C_conv3
- Depthwise Conv → C_dw
```

**실험 설계:**
```python
# Vision task로 측정 (spatial structure 필요)
# Input: [B, C, H, W]

class ConvLayer:
    def __init__(self, c_in, c_out, conv_type):
        if conv_type == 'dense':
            self.op = Linear(c_in * H * W, c_out * H * W)
        elif conv_type == 'conv1x1':
            self.op = Conv2d(c_in, c_out, kernel_size=1)
        elif conv_type == 'conv3x3':
            self.op = Conv2d(c_in, c_out, kernel_size=3, padding=1)
        elif conv_type == 'depthwise':
            self.op = Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in)
```

**실험량:** 4종 × 3 seeds = 12 runs

---

### 1C. Normalization 상수

**목표:** 정규화 기법별 정보 보존 비율

**실험 구조:**
```
기준: No Normalization = 1.000

측정 대상:
- LayerNorm → C_ln
- RMSNorm → C_rms
- BatchNorm → C_bn
- GroupNorm → C_gn
```

**실험 설계:**
```python
class NormLayer:
    def __init__(self, d, norm_type):
        self.linear = Linear(d, d)
        self.norm = get_norm(norm_type, d)
        self.act = ReLU()  # 고정
    
    def forward(self, x):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        return self.act(x)
```

**실험량:** 5종 × 3 seeds = 15 runs

---

### 1D. Precision 상수

**목표:** 수치 정밀도별 정보 손실 비율

**실험 구조:**
```
기준: FP32 = 1.000

측정 대상:
- FP16 → C_fp16
- BF16 → C_bf16
- INT8 → C_int8
- INT4 → C_int4
```

**실험 설계:**
```python
# FP32로 학습 → 각 precision으로 변환 → κ 측정

net = train_fp32(data)
κ_fp32 = compute_kappa(net, precision='fp32')

for prec in ['fp16', 'bf16', 'int8', 'int4']:
    net_quantized = quantize(net, precision=prec)
    κ_prec = compute_kappa(net_quantized)
    C_prec = κ_prec / κ_fp32
```

**실험량:** 5종 × 3 seeds = 15 runs

---

### 1E. Skip Connection 상수

**목표:** Skip connection 유형별 정보 증폭 비율

**실험 구조:**
```
기준: No Skip = 1.000

측정 대상:
- Residual Skip (x + f(x)) → C_skip
- Dense Skip (concat) → C_dense_skip
```

**실험 설계:**
```python
class SkipBlock:
    def __init__(self, d, skip_type):
        self.linear = Linear(d, d)
        self.act = ReLU()
        self.skip_type = skip_type
    
    def forward(self, x):
        fx = self.act(self.linear(x))
        
        if self.skip_type == 'none':
            return fx
        elif self.skip_type == 'residual':
            return x + fx
        elif self.skip_type == 'dense':
            return torch.cat([x, fx], dim=-1)
```

**실험량:** 3종 × 3 seeds = 9 runs

---

### 1F. 독립성 검증 (Critical)

**목표:** 상수들이 곱셈으로 조합되는지 검증

**핵심 가설:**
```
κ_total = κ_base × C_act × C_norm × C_prec × C_skip

예측: κ(GELU + LN + FP16) = κ_relu × C_gelu × C_ln × C_fp16
```

**실험 설계:**
```python
# 1단계: 개별 상수 측정 (1A-1E에서 완료)
C = {
    'gelu': 1.070, 'silu': 1.044,
    'ln': 0.990, 'rms': 0.992,
    'fp16': 0.980, 'int8': 0.920,
    'skip': 1.020
}

# 2단계: 조합 예측
def predict_kappa(config):
    κ_pred = κ_base
    for component, value in config.items():
        κ_pred *= C[component]
    return κ_pred

# 3단계: 조합 실측
test_configs = [
    {'act': 'gelu', 'norm': 'ln', 'prec': 'fp16'},
    {'act': 'silu', 'norm': 'rms', 'prec': 'int8'},
    {'act': 'gelu', 'norm': 'bn', 'skip': 'residual'},
    # ... 15개 조합
]

for config in test_configs:
    net = build_net(config)
    κ_measured = compute_kappa(net)
    κ_predicted = predict_kappa(config)
    error = abs(κ_measured - κ_predicted) / κ_measured
    
# 검증 기준: 평균 오차 < 5%
```

**테스트 조합 (15개):**
```
1. GELU + LN + FP32
2. GELU + LN + FP16
3. GELU + RMSNorm + FP32
4. SiLU + LN + FP32
5. SiLU + RMSNorm + INT8
6. ReLU + BN + FP32 + Skip
7. GELU + LN + FP16 + Skip
8. Tanh + GN + FP32
9. Sigmoid + LN + FP16
10. Mish + RMSNorm + BF16
11. GELU + BN + INT8
12. SiLU + LN + INT4
13. ReLU + None + FP32 + Skip
14. GELU + GN + BF16 + Skip
15. Identity + LN + FP32
```

**실험량:** 15종 × 3 seeds = 45 runs

**성공 기준:**
- 개별 오차: 모두 < 10%
- 평균 오차: < 5%
- 최대 오차: < 15%

---

### 1G. 교차 검증 (Robustness)

**목표:** 상수가 조건 변화에도 안정적인지 검증

**검증 축:**

**1. 데이터셋 변화**
```
- MNIST (28×28, grayscale)
- CIFAR-10 (32×32, color)
- ImageNet subset (224×224, color)
```

**2. MI 추정 방법 변화**
```
- KSG estimator
- MINE estimator
- InfoNCE estimator
```

**3. Width 변화**
```
- d = 256
- d = 512
- d = 1024
```

**실험 설계:**
```python
# GELU 상수가 조건마다 일정한지 검증
conditions = [
    ('mnist', 'ksg', 256),
    ('mnist', 'mine', 256),
    ('cifar', 'ksg', 256),
    ('cifar', 'ksg', 512),
    ('imagenet', 'ksg', 256),
    # ... 12개 조합
]

C_gelu_list = []
for dataset, mi_method, width in conditions:
    κ_gelu = measure(dataset, mi_method, width, act='gelu')
    κ_relu = measure(dataset, mi_method, width, act='relu')
    C_gelu_list.append(κ_gelu / κ_relu)

# 검증: std(C_gelu_list) < 0.03
```

**실험량:** 12종 × 3 seeds = 36 runs

**성공 기준:**
- 각 상수의 조건 간 변동: std < 3%
- 극단 조건에서도 비율 유지

---

### 1편 총 실험량

| 섹션 | 실험 |
|------|------|
| 1A Activation | 21 runs |
| 1B Linear Type | 12 runs |
| 1C Normalization | 15 runs |
| 1D Precision | 15 runs |
| 1E Skip | 9 runs |
| 1F 독립성 | 45 runs |
| 1G 교차검증 | 36 runs |
| **Total** | **~153 runs** |

---

### 1편 산출물

```yaml
# ICON-Primitive Constants v1.0

activation:
  relu: 1.000      # 기준
  gelu: 1.070 ± 0.015
  silu: 1.044 ± 0.012
  tanh: 0.933 ± 0.018
  sigmoid: 0.890 ± 0.020
  mish: 1.055 ± 0.014
  identity: 1.150 ± 0.025

linear_type:
  dense: 1.000     # 기준
  conv1x1: 0.995 ± 0.010
  conv3x3: 0.980 ± 0.012
  depthwise: 0.960 ± 0.015

normalization:
  none: 1.000      # 기준
  layernorm: 0.990 ± 0.008
  rmsnorm: 0.992 ± 0.007
  batchnorm: 0.985 ± 0.010
  groupnorm: 0.988 ± 0.009

precision:
  fp32: 1.000      # 기준
  fp16: 0.980 ± 0.010
  bf16: 0.975 ± 0.012
  int8: 0.920 ± 0.018
  int4: 0.800 ± 0.030

skip:
  none: 1.000      # 기준
  residual: 1.020 ± 0.008
  dense: 1.035 ± 0.010

# 조합 공식
formula: "κ_rel = C_act × C_type × C_norm × C_prec × C_skip"
independence_verified: true
cross_validation_passed: true
```

---

### 1편 논문 구조

```
Abstract
1. Introduction
   - 문제: 모델 효율성 측정의 부재
   - 기여: 기초 연산 상수 제안

2. Background
   - Information Capacity (κ) 정의
   - Mutual Information 추정 방법

3. Method
   - 상대 비율 기반 상수 정의
   - 측정 프로토콜

4. Experiments
   - 4.1 Activation Constants
   - 4.2 Linear Type Constants
   - 4.3 Normalization Constants
   - 4.4 Precision Constants
   - 4.5 Skip Connection Constants
   - 4.6 Independence Verification
   - 4.7 Cross-Validation

5. Results
   - 상수 테이블
   - 독립성 검증 결과
   - 교차검증 결과

6. Discussion
   - 상수의 의미 해석
   - 한계 및 future work

7. Conclusion

Appendix A: MI 추정 상세
Appendix B: 전체 실험 결과
Appendix C: 코드 및 재현성
```

---
---

# 2편: ICON-Block

## "복합 블록의 정보 용량 상수"

---

### 핵심 주장

> 복합 블록의 정보 용량 상수는 구성 요소의 함수로 표현되며,
> 블록 수준에서도 상수 조합이 성립한다.

---

### 2A. FFN 계열 상수

**목표:** Feed-Forward Network 변형들의 정보 용량 비율

**실험 구조:**
```
기준: FFN-ReLU (Linear → ReLU → Linear) = 1.000

측정 대상:
- FFN-GELU → C_ffn_gelu
- GLU (σ(Wx) ⊙ Vx) → C_glu
- SwiGLU (SiLU(Wx) ⊙ Vx) → C_swiglu
- GeGLU (GELU(Wx) ⊙ Vx) → C_geglu
- ReGLU (ReLU(Wx) ⊙ Vx) → C_reglu
```

**실험 설계:**
```python
class FFN:
    def __init__(self, d_model, d_ff, ffn_type):
        if ffn_type == 'ffn_relu':
            self.net = Sequential(
                Linear(d_model, d_ff),
                ReLU(),
                Linear(d_ff, d_model)
            )
        elif ffn_type == 'swiglu':
            self.W = Linear(d_model, d_ff)
            self.V = Linear(d_model, d_ff)
            self.out = Linear(d_ff, d_model)
            
    def forward(self, x):
        if self.ffn_type == 'swiglu':
            return self.out(F.silu(self.W(x)) * self.V(x))
        else:
            return self.net(x)
```

**검증:** `C_ffn = f(C_act)` 관계 찾기
```
예상:
C_ffn_gelu ≈ C_gelu^α (α ≈ 1.0-1.5)
C_swiglu ≈ C_silu × C_gating_factor
```

**실험량:** 6종 × 3 seeds = 18 runs

---

### 2B. Attention 계열 상수

**목표:** Attention 메커니즘 변형들의 정보 용량 비율

**실험 구조:**
```
기준: Single-Head Attention = 1.000

측정 대상:
- Multi-Head Attention (8 heads) → C_mha
- Grouped Query Attention → C_gqa
- Multi-Query Attention → C_mqa
- Linear Attention → C_linear_attn
- Sparse Attention (local window) → C_sparse
```

**실험 설계:**
```python
class Attention:
    def __init__(self, d_model, n_heads, attn_type):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        if attn_type == 'mha':
            self.qkv = Linear(d_model, 3 * d_model)
        elif attn_type == 'gqa':
            self.n_kv_heads = n_heads // 4
            self.q = Linear(d_model, d_model)
            self.kv = Linear(d_model, 2 * self.n_kv_heads * self.d_head)
        elif attn_type == 'mqa':
            self.q = Linear(d_model, d_model)
            self.kv = Linear(d_model, 2 * self.d_head)
        elif attn_type == 'linear':
            # φ(Q) × (φ(K)^T × V)
            self.feature_map = lambda x: F.elu(x) + 1
```

**실험량:** 6종 × 3 seeds = 18 runs

---

### 2C. Conv 블록 상수

**목표:** CNN 블록 변형들의 정보 용량 비율

**실험 구조:**
```
기준: Basic Conv Block (Conv → Norm → Act) = 1.000

측정 대상:
- Bottleneck (1×1 → 3×3 → 1×1) → C_bottleneck
- Inverted Residual (1×1 → DW 3×3 → 1×1) → C_inverted
- SE Block (Squeeze-Excite) → C_se
- ConvNeXt Block (DW → LN → FFN) → C_convnext
```

**실험 설계:**
```python
class ConvBlock:
    def __init__(self, channels, block_type):
        if block_type == 'basic':
            self.net = Sequential(
                Conv2d(channels, channels, 3, padding=1),
                BatchNorm2d(channels),
                ReLU()
            )
        elif block_type == 'bottleneck':
            mid = channels // 4
            self.net = Sequential(
                Conv2d(channels, mid, 1),
                BatchNorm2d(mid),
                ReLU(),
                Conv2d(mid, mid, 3, padding=1),
                BatchNorm2d(mid),
                ReLU(),
                Conv2d(mid, channels, 1),
                BatchNorm2d(channels)
            )
        elif block_type == 'se':
            self.conv = Conv2d(channels, channels, 3, padding=1)
            self.se = Sequential(
                AdaptiveAvgPool2d(1),
                Linear(channels, channels // 16),
                ReLU(),
                Linear(channels // 16, channels),
                Sigmoid()
            )
```

**실험량:** 5종 × 3 seeds = 15 runs

---

### 2D. Aggregation 상수 (GNN용)

**목표:** 그래프 집계 연산의 정보 용량 비율

**실험 구조:**
```
기준: Sum Aggregation = 1.000

측정 대상:
- Mean Aggregation → C_mean
- Max Aggregation → C_max
- Attention Aggregation → C_attn_agg
```

**실험 설계:**
```python
class GNNLayer:
    def __init__(self, d, agg_type):
        self.linear = Linear(d, d)
        self.agg_type = agg_type
        
    def forward(self, x, edge_index):
        # x: [N, d], edge_index: [2, E]
        src, dst = edge_index
        messages = self.linear(x[src])
        
        if self.agg_type == 'sum':
            out = scatter_sum(messages, dst, dim=0)
        elif self.agg_type == 'mean':
            out = scatter_mean(messages, dst, dim=0)
        elif self.agg_type == 'max':
            out = scatter_max(messages, dst, dim=0)
        elif self.agg_type == 'attention':
            # GAT style
            attn_weights = compute_attention(x, edge_index)
            out = scatter_sum(messages * attn_weights, dst, dim=0)
```

**데이터셋:** Cora, CiteSeer, PubMed

**실험량:** 4종 × 3 seeds = 12 runs

---

### 2E. Recurrent 상수

**목표:** 순환 구조의 정보 용량 비율

**실험 구조:**
```
기준: Vanilla RNN = 1.000

측정 대상:
- LSTM → C_lstm
- GRU → C_gru
- SSM (Mamba-style) → C_ssm
```

**실험 설계:**
```python
class RecurrentBlock:
    def __init__(self, d, rnn_type):
        if rnn_type == 'vanilla':
            self.rnn = RNN(d, d, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = LSTM(d, d, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = GRU(d, d, batch_first=True)
        elif rnn_type == 'ssm':
            # Simplified S4/Mamba
            self.A = nn.Parameter(torch.randn(d, d))
            self.B = nn.Parameter(torch.randn(d, d))
            self.C = nn.Parameter(torch.randn(d, d))
            
    def forward(self, x):
        # x: [B, T, d]
        if self.rnn_type == 'ssm':
            # h_t = A @ h_{t-1} + B @ x_t
            # y_t = C @ h_t
            return ssm_forward(x, self.A, self.B, self.C)
        else:
            return self.rnn(x)[0]
```

**실험량:** 4종 × 3 seeds = 12 runs

---

### 2F. Depth 함수

**목표:** 깊이에 따른 정보 용량 변화 패턴

**측정:**
```
Depth = 1, 2, 4, 8, 16, 32

가설:
- 지수 감쇠: κ(n) = β^(n-1)
- 선형 감쇠: κ(n) = 1 - α(n-1)
- 로그 감쇠: κ(n) = 1 / log(n+1)
```

**실험 설계:**
```python
def measure_depth_scaling():
    results = {}
    
    for depth in [1, 2, 4, 8, 16, 32]:
        # 같은 total params를 유지하기 위해 width 조절
        width = base_params / depth
        
        net = StackedMLP(
            d_in=256,
            d_hidden=width,
            d_out=256,
            depth=depth,
            activation=ReLU()
        )
        train(net, data)
        κ = compute_kappa(net)
        results[depth] = κ
    
    # 패턴 fitting
    # κ(n) / κ(1) vs n
    ratios = [results[d] / results[1] for d in depths]
    
    # Fit: ratio = β^(n-1)
    β = curve_fit(exponential_decay, depths, ratios)
```

**실험량:** 6종 × 3 seeds = 18 runs

**예상 결과:**
```
β ≈ 0.995-0.999 (매우 느린 감쇠)
→ 깊이 32에서도 ~85-95% 유지
```

---

### 2G. Layer 조합 상수

**목표:** 완전한 레이어(여러 블록 조합)의 정보 용량

**실험 구조:**
```
측정 대상:
- Transformer Layer (Attn + FFN + Skip + LN)
- LLaMA Layer (Attn + SwiGLU + Skip + RMSNorm)
- ResNet Block (Conv + Conv + Skip + BN)
- ConvNeXt Layer (DW + FFN + Skip + LN)
- Mamba Layer (SSM + Linear + Skip)
- GNN Layer (Aggregate + Linear + Act)
```

**실험 설계:**
```python
class TransformerLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff, 'gelu')
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LLaMALayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = GroupedQueryAttention(d_model, n_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
```

**검증:** `C_layer = f(C_attn, C_ffn, C_skip, C_norm)` 공식 도출

```
예상 공식:
C_transformer ≈ C_mha × C_ffn_gelu × C_skip^2 × C_ln^2

검증:
예측 = 1.05 × 0.96 × 1.02^2 × 0.99^2 = 1.007
실측 = ?
```

**실험량:** 6종 × 3 seeds = 18 runs

---

### 2H. 블록 독립성 검증

**목표:** 레이어 상수가 블록 상수의 조합인지 검증

**실험 설계:**
```python
# 1단계: 개별 블록 상수 (2A-2E에서 측정)
C_block = {
    'mha': 1.050,
    'ffn_gelu': 0.960,
    'swiglu': 1.020,
    'skip': 1.020,
    'ln': 0.990,
    'rms': 0.992,
}

# 2단계: 조합 공식으로 예측
def predict_layer_kappa(layer_type):
    if layer_type == 'transformer':
        return C_block['mha'] * C_block['ffn_gelu'] * C_block['skip']**2 * C_block['ln']**2
    elif layer_type == 'llama':
        return C_block['gqa'] * C_block['swiglu'] * C_block['skip']**2 * C_block['rms']**2
    # ...

# 3단계: 실측과 비교
for layer_type in layer_types:
    predicted = predict_layer_kappa(layer_type)
    measured = measure_layer_kappa(layer_type)
    error = abs(predicted - measured) / measured
```

**테스트 케이스:**
```
1. Transformer Layer
2. LLaMA Layer
3. Pre-LN Transformer
4. Post-LN Transformer
5. ResNet Block
6. ResNeXt Block
7. ConvNeXt Layer
8. Mamba Layer
9. GPT-style Layer
10. BERT-style Layer
```

**실험량:** 10종 × 3 seeds = 30 runs

**성공 기준:**
- 평균 오차 < 7%
- 최대 오차 < 15%

---

### 2편 총 실험량

| 섹션 | 실험 |
|------|------|
| 2A FFN | 18 runs |
| 2B Attention | 18 runs |
| 2C Conv Block | 15 runs |
| 2D Aggregation | 12 runs |
| 2E Recurrent | 12 runs |
| 2F Depth | 18 runs |
| 2G Layer | 18 runs |
| 2H 독립성 | 30 runs |
| **Total** | **~141 runs** |

---

### 2편 산출물

```yaml
# ICON-Block Constants v1.0

ffn:
  ffn_relu: 1.000
  ffn_gelu: 0.960 ± 0.012
  glu: 0.970 ± 0.015
  swiglu: 1.020 ± 0.010
  geglu: 1.010 ± 0.012
  reglu: 0.965 ± 0.014

attention:
  single_head: 1.000
  multi_head_8: 1.050 ± 0.015
  grouped_query: 1.040 ± 0.014
  multi_query: 1.030 ± 0.012
  linear: 0.920 ± 0.020
  sparse: 0.950 ± 0.018

conv_block:
  basic: 1.000
  bottleneck: 0.970 ± 0.015
  inverted_residual: 0.985 ± 0.012
  se_block: 1.010 ± 0.014
  convnext: 1.000 ± 0.010

aggregation:
  sum: 1.000
  mean: 0.980 ± 0.015
  max: 0.950 ± 0.020
  attention: 1.050 ± 0.018

recurrent:
  vanilla_rnn: 1.000
  lstm: 1.050 ± 0.020
  gru: 1.040 ± 0.018
  ssm: 1.080 ± 0.025

depth:
  formula: "C(n) = β^(n-1)"
  beta: 0.998 ± 0.001

layer:
  transformer: 0.950 ± 0.015
  llama: 0.970 ± 0.012
  resnet: 0.940 ± 0.018
  convnext: 0.960 ± 0.014
  mamba: 0.980 ± 0.015
  gnn: 0.930 ± 0.020

# 조합 공식
block_composition: "C_layer = C_main × C_aux × C_skip^n_skip × C_norm^n_norm"
independence_verified: true
```

---

### 2편 논문 구조

```
Abstract
1. Introduction
   - 1편 요약: 기초 상수
   - 2편 목표: 블록 수준으로 확장

2. Background
   - 1편 상수 recap
   - 블록 정의

3. Method
   - 블록별 κ 측정 방법
   - 조합 공식 도출 방법

4. Experiments
   - 4.1 FFN Block Constants
   - 4.2 Attention Constants
   - 4.3 Conv Block Constants
   - 4.4 Aggregation Constants
   - 4.5 Recurrent Constants
   - 4.6 Depth Scaling
   - 4.7 Layer Composition
   - 4.8 Block Independence

5. Results
   - 블록 상수 테이블
   - Depth 함수
   - 조합 공식

6. Discussion
   - 1편 상수 → 2편 블록 관계
   - 아키텍처 설계 시사점

7. Conclusion

Appendix A: 블록 구현 상세
Appendix B: 전체 결과
```

---
---

# 3편: ICON-Paradigm

## "학습 패러다임별 정보 용량 상수"

---

### 핵심 주장

> 학습 패러다임마다 정보 용량 정의가 달라지며,
> 패러다임 내에서 상수 비율이 안정적으로 유지된다.

---

### 3A. 결정론적 (Deterministic) - 기준

**이미 1-2편에서 커버:**
```
- MLP (1편)
- CNN (2편 Conv Block)
- Transformer (2편 Layer)
- Mamba (2편 Recurrent)
- GNN (2편 Aggregation)
```

**κ 정의:**
```
κ_det = I(X; Z) / d_z
"입력이 출력에 얼마나 전달되는가"
```

---

### 3B. 확률론적 (Probabilistic)

**새로운 κ 정의:**
```
κ_prob = I(X; μ) / d_latent
"입력이 분포의 평균에 얼마나 인코딩되는가"
```

**측정 대상:**
```
1. VAE Encoder: x → (μ, σ)
2. VAE Sampling: z = μ + σ·ε
3. VAE Full: Encoder → Sampling → Decoder
4. Bayesian Layer: p(W|X) 기반
5. MC Dropout: 추론 시 dropout
```

**실험 설계:**
```python
class VAE:
    def __init__(self, d_in, d_latent):
        self.encoder = Sequential(
            Linear(d_in, 256),
            ReLU(),
            Linear(256, 128),
            ReLU()
        )
        self.mu = Linear(128, d_latent)
        self.logvar = Linear(128, d_latent)
        self.decoder = Sequential(
            Linear(d_latent, 128),
            ReLU(),
            Linear(128, 256),
            ReLU(),
            Linear(256, d_in)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        μ = self.mu(h)
        σ = torch.exp(0.5 * self.logvar(h))
        z = μ + σ * torch.randn_like(σ)
        return self.decoder(z), μ, σ

# κ 측정
def compute_kappa_vae_encoder(vae, data):
    μ_list = []
    for x in data:
        _, μ, _ = vae(x)
        μ_list.append(μ)
    
    # I(X; μ) / d_latent
    return estimate_mi(data, torch.stack(μ_list)) / d_latent

def compute_kappa_vae_full(vae, data):
    z_list = []
    for x in data:
        h = vae.encoder(x)
        μ = vae.mu(h)
        σ = torch.exp(0.5 * vae.logvar(h))
        z = μ + σ * torch.randn_like(σ)
        z_list.append(z)
    
    return estimate_mi(data, torch.stack(z_list)) / d_latent
```

**Bayesian Layer:**
```python
class BayesianLinear:
    def __init__(self, d_in, d_out):
        self.W_mu = nn.Parameter(torch.randn(d_out, d_in))
        self.W_rho = nn.Parameter(torch.zeros(d_out, d_in))
    
    def forward(self, x):
        σ = F.softplus(self.W_rho)
        W = self.W_mu + σ * torch.randn_like(σ)
        return F.linear(x, W)
```

**MC Dropout:**
```python
class MCDropoutNet:
    def __init__(self, d, p=0.1):
        self.linear = Linear(d, d)
        self.dropout = Dropout(p)
        # 추론 시에도 dropout 유지
        
    def forward(self, x):
        return self.dropout(self.linear(x))
    
    def predict_with_uncertainty(self, x, n_samples=30):
        preds = [self.forward(x) for _ in range(n_samples)]
        return torch.stack(preds).mean(0), torch.stack(preds).std(0)
```

**실험량:** 5종 × 3 seeds = 15 runs

---

### 3C. 생성적 (Generative)

**새로운 κ 정의:**
```
κ_gen = I(Z_input; X_output) / d_output
"입력 노이즈/조건이 출력에 얼마나 전달되는가"
```

**측정 대상:**
```
1. GAN Generator: z → x_fake
2. GAN Discriminator: x → real/fake
3. Diffusion Forward: x → x_t (noise 추가)
4. Diffusion Reverse: x_t → ε_pred (노이즈 예측)
5. Flow: x ↔ z (bijective)
6. Autoregressive: x_{<t} → x_t
```

**실험 설계:**
```python
# GAN Generator
class Generator:
    def __init__(self, d_z, d_out):
        self.net = Sequential(
            Linear(d_z, 256),
            ReLU(),
            Linear(256, 512),
            ReLU(),
            Linear(512, d_out),
            Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

def compute_kappa_generator(gen, n_samples=10000):
    z = torch.randn(n_samples, d_z)
    x = gen(z)
    # I(Z; X) / d_out
    return estimate_mi(z, x) / d_out

# Diffusion
class DiffusionModel:
    def __init__(self, d, n_steps=1000):
        self.denoise_net = UNet(d)
        self.n_steps = n_steps
        self.betas = linear_schedule(n_steps)
    
    def forward_process(self, x, t):
        # x_t = sqrt(α_bar) * x + sqrt(1 - α_bar) * ε
        noise = torch.randn_like(x)
        α_bar = self.compute_alpha_bar(t)
        return torch.sqrt(α_bar) * x + torch.sqrt(1 - α_bar) * noise, noise
    
    def reverse_predict(self, x_t, t):
        return self.denoise_net(x_t, t)

def compute_kappa_diffusion_forward(model, data, t):
    x_t, noise = model.forward_process(data, t)
    # I(X; X_t) / d
    return estimate_mi(data, x_t) / d

def compute_kappa_diffusion_reverse(model, data, t):
    x_t, noise = model.forward_process(data, t)
    ε_pred = model.reverse_predict(x_t, t)
    # I(X_t; ε_pred) / d
    return estimate_mi(x_t, ε_pred) / d

# Flow (RealNVP style)
class FlowBlock:
    def __init__(self, d):
        self.s = MLP(d // 2, d // 2)
        self.t = MLP(d // 2, d // 2)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        s, t = self.s(x1), self.t(x1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=-1)
    
    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=-1)
        s, t = self.s(y1), self.t(y1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=-1)

# Autoregressive
class AutoregressiveModel:
    def __init__(self, vocab_size, d_model, n_layers):
        self.embed = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model) for _ in range(n_layers)
        ])
        self.head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: [B, T] token indices
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)  # [B, T, vocab]

def compute_kappa_autoregressive(model, data):
    # I(X_{<t}; X_t) / d
    # Context가 다음 토큰 예측에 얼마나 기여하는가
    pass
```

**실험량:** 6종 × 3 seeds = 18 runs

---

### 3D. 이산적 (Discrete/Spiking)

**새로운 κ 정의:**
```
κ_spike = I(X; Spike_train) / (d × T)
"시공간 정보 용량"
```

**측정 대상:**
```
1. SNN (LIF neuron): Leaky Integrate-and-Fire
2. SNN (IF neuron): Integrate-and-Fire
3. Binary NN: 1-bit weights
4. Ternary NN: {-1, 0, 1} weights
```

**실험 설계:**
```python
# LIF Neuron
class LIFNeuron:
    def __init__(self, tau=20.0, v_th=1.0, v_reset=0.0):
        self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset
    
    def forward(self, x, v):
        # x: [B, T, d] input current
        # v: [B, d] membrane potential
        spikes = []
        for t in range(x.size(1)):
            v = v * (1 - 1/self.tau) + x[:, t]
            spike = (v >= self.v_th).float()
            v = v * (1 - spike) + self.v_reset * spike
            spikes.append(spike)
        return torch.stack(spikes, dim=1)  # [B, T, d]

class SNNLayer:
    def __init__(self, d_in, d_out):
        self.linear = Linear(d_in, d_out)
        self.lif = LIFNeuron()
    
    def forward(self, x):
        # x: [B, T, d_in] spike train
        current = self.linear(x)
        v = torch.zeros(x.size(0), d_out)
        return self.lif(current, v)

def compute_kappa_snn(model, data, T):
    # data: [B, d] static input → encode to spikes
    input_spikes = rate_encode(data, T)  # [B, T, d]
    output_spikes = model(input_spikes)
    
    # I(X; output_spikes) / (d × T)
    return estimate_mi(data, output_spikes.flatten(1)) / (d_out * T)

# Binary NN
class BinaryLinear:
    def __init__(self, d_in, d_out):
        self.weight = nn.Parameter(torch.randn(d_out, d_in))
    
    def forward(self, x):
        # Binarize weights: sign(W)
        W_binary = torch.sign(self.weight)
        return F.linear(x, W_binary)

# Ternary NN
class TernaryLinear:
    def __init__(self, d_in, d_out, threshold=0.5):
        self.weight = nn.Parameter(torch.randn(d_out, d_in))
        self.threshold = threshold
    
    def forward(self, x):
        # Ternarize: {-1, 0, 1}
        W = self.weight
        W_ternary = torch.zeros_like(W)
        W_ternary[W > self.threshold] = 1
        W_ternary[W < -self.threshold] = -1
        return F.linear(x, W_ternary)
```

**실험량:** 4종 × 3 seeds = 12 runs

---

### 3E. 동적 (Dynamic)

**새로운 κ 정의:**
```
κ_dynamic = E[I(X; Z)] / d_z
"기대 정보 용량"
(입력에 따라 경로가 달라지므로 기대값)
```

**측정 대상:**
```
1. MoE Routing: Expert 선택
2. MoE Expert: 선택된 expert 연산
3. Early Exit: 조건부 탈출
4. Adaptive Depth: 동적 깊이
5. Dynamic Conv: 입력 따라 kernel 변화
```

**실험 설계:**
```python
# Mixture of Experts
class MoE:
    def __init__(self, d, n_experts, top_k=2):
        self.router = Linear(d, n_experts)
        self.experts = nn.ModuleList([
            FFN(d, d * 4) for _ in range(n_experts)
        ])
        self.top_k = top_k
    
    def forward(self, x):
        # x: [B, d]
        logits = self.router(x)  # [B, n_experts]
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = probs.topk(self.top_k, dim=-1)
        
        # 선택된 experts만 실행
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = top_k_probs[mask, (top_k_indices[mask] == i).nonzero()[:, 1]]
                output[mask] += weight.unsqueeze(-1) * expert_out
        
        return output

def compute_kappa_moe(model, data, n_samples=100):
    # 각 샘플마다 선택되는 expert가 다름
    # → 기대값 계산
    κ_list = []
    for _ in range(n_samples):
        subset = data[torch.randperm(len(data))[:1000]]
        out = model(subset)
        κ = estimate_mi(subset, out) / d
        κ_list.append(κ)
    
    return np.mean(κ_list)

# Early Exit
class EarlyExitNet:
    def __init__(self, d, n_layers, threshold=0.9):
        self.layers = nn.ModuleList([
            TransformerLayer(d) for _ in range(n_layers)
        ])
        self.classifiers = nn.ModuleList([
            Linear(d, n_classes) for _ in range(n_layers)
        ])
        self.threshold = threshold
    
    def forward(self, x):
        for i, (layer, clf) in enumerate(zip(self.layers, self.classifiers)):
            x = layer(x)
            logits = clf(x)
            confidence = F.softmax(logits, dim=-1).max(dim=-1).values
            
            if confidence.mean() > self.threshold:
                return x, i  # Early exit at layer i
        
        return x, len(self.layers) - 1

# Adaptive Depth
class AdaptiveDepthNet:
    def __init__(self, d, max_depth):
        self.layers = nn.ModuleList([
            TransformerLayer(d) for _ in range(max_depth)
        ])
        self.halting = nn.ModuleList([
            Linear(d, 1) for _ in range(max_depth)
        ])
    
    def forward(self, x):
        # ACT (Adaptive Computation Time) style
        halting_prob = torch.zeros(x.size(0))
        remainders = torch.ones(x.size(0))
        n_updates = torch.zeros(x.size(0))
        
        for layer, halt in zip(self.layers, self.halting):
            p = torch.sigmoid(halt(x)).squeeze(-1)
            still_running = (halting_prob < 1.0).float()
            
            new_halting = halting_prob + p * still_running
            # ... ACT 로직
            
            x = layer(x)
        
        return x
```

**실험량:** 5종 × 3 seeds = 15 runs

---

### 3F. 패러다임 간 비율

**목표:** 서로 다른 패러다임 상수를 비교할 수 있게 정규화

**방법:**
```
기준: Deterministic (MLP with same architecture) = 1.000

각 패러다임에서:
1. 동일 구조의 deterministic 버전 만듦
2. 해당 패러다임 버전 만듦
3. 비율 계산

예:
- VAE vs Deterministic Autoencoder
- GAN Generator vs Deterministic Generator (MSE 학습)
- SNN vs 동일 구조 ANN
```

**실험 설계:**
```python
def compute_paradigm_ratio(paradigm, architecture):
    # Deterministic baseline
    det_model = build_deterministic(architecture)
    train_deterministic(det_model, data)
    κ_det = compute_kappa_det(det_model)
    
    # Paradigm-specific
    if paradigm == 'probabilistic':
        model = build_vae(architecture)
        train_vae(model, data)
        κ_paradigm = compute_kappa_prob(model)
    elif paradigm == 'generative':
        model = build_gan(architecture)
        train_gan(model, data)
        κ_paradigm = compute_kappa_gen(model)
    # ...
    
    return κ_paradigm / κ_det
```

**테스트 케이스:**
```
Probabilistic:
1. VAE vs Det-AE
2. Bayesian MLP vs Det-MLP

Generative:
3. GAN Gen vs Det-Gen
4. Diffusion vs Det-Denoise
5. Flow vs Det-Transform

Spiking:
6. SNN-LIF vs ANN
7. Binary NN vs Full-precision

Dynamic:
8. MoE vs Dense FFN
9. Early Exit vs Full Depth
10. Adaptive vs Fixed
```

**실험량:** 10종 × 3 seeds = 30 runs

---

### 3G. 교차 검증

**목표:** 패러다임 상수가 조건 변화에 안정적인지

**검증 축:**
```
1. 아키텍처 변화
   - Small (256-dim)
   - Medium (512-dim)
   - Large (1024-dim)

2. 데이터셋 변화
   - MNIST
   - CIFAR-10
   - CelebA (생성용)

3. 학습 설정 변화
   - Learning rate
   - Batch size
   - Training epochs
```

**실험 설계:**
```python
conditions = [
    ('mnist', 'small', 'default'),
    ('mnist', 'medium', 'default'),
    ('cifar', 'small', 'default'),
    ('cifar', 'medium', 'high_lr'),
    ('celeba', 'medium', 'default'),
    # ... 15개 조합
]

C_vae_list = []
for dataset, arch, setting in conditions:
    # VAE 상수 측정
    vae = train_vae(dataset, arch, setting)
    det = train_det_ae(dataset, arch, setting)
    
    κ_vae = compute_kappa_prob(vae)
    κ_det = compute_kappa_det(det)
    
    C_vae_list.append(κ_vae / κ_det)

# 검증: std(C_vae_list) < 5%
```

**실험량:** 15종 × 3 seeds = 45 runs

---

### 3편 총 실험량

| 섹션 | 실험 |
|------|------|
| 3B Probabilistic | 15 runs |
| 3C Generative | 18 runs |
| 3D Spiking | 12 runs |
| 3E Dynamic | 15 runs |
| 3F 패러다임 비율 | 30 runs |
| 3G 교차검증 | 45 runs |
| **Total** | **~135 runs** |

---

### 3편 산출물

```yaml
# ICON-Paradigm Constants v1.0

# κ 정의
definitions:
  deterministic: "κ = I(X; Z) / d_z"
  probabilistic: "κ = I(X; μ) / d_latent"
  generative: "κ = I(Z_in; X_out) / d_out"
  spiking: "κ = I(X; Spikes) / (d × T)"
  dynamic: "κ = E[I(X; Z)] / d_z"

# 기준 대비 상수 (Deterministic = 1.000)
paradigm_constants:
  deterministic: 1.000
  
  probabilistic:
    vae_encoder: 0.920 ± 0.020
    vae_sampling: 0.850 ± 0.025
    vae_full: 0.780 ± 0.030
    bayesian_layer: 0.880 ± 0.022
    mc_dropout: 0.950 ± 0.015
  
  generative:
    gan_generator: 0.820 ± 0.030
    gan_discriminator: 0.900 ± 0.020
    diffusion_forward: 0.950 ± 0.015
    diffusion_reverse: 0.880 ± 0.025
    flow: 0.920 ± 0.020
    autoregressive: 0.940 ± 0.018
  
  spiking:
    lif: 0.750 ± 0.035
    if: 0.720 ± 0.040
    binary_nn: 0.650 ± 0.045
    ternary_nn: 0.750 ± 0.035
  
  dynamic:
    moe_routing: 0.980 ± 0.012
    moe_expert: 1.050 ± 0.018
    early_exit: 0.990 ± 0.010
    adaptive_depth: 0.970 ± 0.015
    dynamic_conv: 0.960 ± 0.018

cross_validation:
  passed: true
  max_std: 0.04
```

---

### 3편 논문 구조

```
Abstract
1. Introduction
   - 1-2편 요약
   - 패러다임별 κ 정의의 필요성

2. Background
   - 기존 상수 (1-2편)
   - 학습 패러다임 분류

3. Method
   - 패러다임별 κ 정의
   - 측정 프로토콜
   - 패러다임 간 비교 방법

4. Experiments
   - 4.1 Probabilistic (VAE, Bayesian)
   - 4.2 Generative (GAN, Diffusion, Flow)
   - 4.3 Spiking (SNN, Binary/Ternary)
   - 4.4 Dynamic (MoE, Early Exit)
   - 4.5 Cross-Paradigm Comparison
   - 4.6 Cross-Validation

5. Results
   - 패러다임별 상수 테이블
   - 교차검증 결과

6. Discussion
   - 패러다임별 정보 효율성 해석
   - 아키텍처 선택 가이드

7. Conclusion

Appendix A: 상세 구현
Appendix B: 추가 결과
Appendix C: κ 정의의 이론적 정당성
```

---
---

# 4편: ICON-Design

## "상수 기반 아키텍처 설계"

---

### 핵심 주장

> ICON 상수를 사용하면 아키텍처의 정보 용량을 **예측**할 수 있고,
> 이를 기반으로 **최적 설계**가 가능하다.

---

### 4A. 설계 원리

**핵심 공식:**
```
목표: 정보량 I_target 달성

필요 차원:
d_required = I_target / κ_total

κ_total = C_act × C_type × C_norm × C_prec × C_skip × C_depth(L) × C_paradigm
```

**설계 흐름:**
```
1. 목표 정보량 설정 (task 기반)
2. 제약 조건 설정 (compute, memory, latency)
3. 상수 기반 κ_total 계산
4. 필요 차원 계산
5. 트레이드오프 분석
6. 최적 구성 선택
```

---

### 4B. 트레이드오프 분석

**핵심 테이블:**
```
| 선택 | κ 변화 | 비용 변화 | 효율 |
|------|--------|-----------|------|
| GELU → ReLU | -7% | -10% FLOPs | +3% |
| FP32 → INT8 | -8% | +2x speed | +92% |
| 8L → 16L | -1.6% | +2x params | -48% |
| Skip 추가 | +2% | +1% params | +1% |
| MHA → GQA | -1% | -25% memory | +24% |
```

**효율 정의:**
```
Efficiency = Δκ / Δcost

Δκ > 0, Δcost < 0 → 최고 (성능↑, 비용↓)
Δκ > 0, Δcost > 0 → 비용 대비 가치 판단
Δκ < 0, Δcost < 0 → 비용 대비 손실 판단
Δκ < 0, Δcost > 0 → 최악 (성능↓, 비용↑)
```

---

### 4C. 최적화 알고리즘

**문제 정의:**
```
maximize κ_total
subject to:
    FLOPs ≤ budget_flops
    Memory ≤ budget_memory
    Latency ≤ budget_latency
```

**알고리즘:**
```python
class ICONOptimizer:
    def __init__(self, constants):
        self.C = constants
    
    def compute_kappa(self, config):
        """아키텍처 config → κ 예측"""
        κ = 1.0
        
        # Primitive 상수
        κ *= self.C['activation'][config['activation']]
        κ *= self.C['normalization'][config['normalization']]
        κ *= self.C['precision'][config['precision']]
        κ *= self.C['skip'][config['skip_type']]
        
        # Block 상수
        κ *= self.C['attention'][config['attention_type']]
        κ *= self.C['ffn'][config['ffn_type']]
        
        # Depth 상수
        κ *= self.C['depth']['beta'] ** (config['n_layers'] - 1)
        
        # Paradigm 상수
        if config['paradigm'] != 'deterministic':
            κ *= self.C['paradigm'][config['paradigm']]
        
        return κ
    
    def compute_cost(self, config):
        """아키텍처 config → 비용 계산"""
        d = config['d_model']
        L = config['n_layers']
        
        flops = 12 * L * d**2  # Transformer 기준 근사
        memory = L * d * 4  # FP32 기준
        
        # Precision 보정
        if config['precision'] == 'fp16':
            memory *= 0.5
        elif config['precision'] == 'int8':
            memory *= 0.25
            flops *= 0.5  # INT8 가속
        
        return {'flops': flops, 'memory': memory}
    
    def optimize(self, I_target, budget):
        """목표 정보량 + 예산 → 최적 설계"""
        best_config = None
        best_efficiency = -float('inf')
        
        # Search space
        activations = ['relu', 'gelu', 'silu']
        norms = ['layernorm', 'rmsnorm']
        precisions = ['fp32', 'fp16', 'int8']
        attn_types = ['mha', 'gqa', 'mqa']
        ffn_types = ['ffn_gelu', 'swiglu']
        depths = [6, 12, 24]
        widths = [256, 512, 768, 1024]
        
        for act in activations:
            for norm in norms:
                for prec in precisions:
                    for attn in attn_types:
                        for ffn in ffn_types:
                            for depth in depths:
                                for width in widths:
                                    config = {
                                        'activation': act,
                                        'normalization': norm,
                                        'precision': prec,
                                        'attention_type': attn,
                                        'ffn_type': ffn,
                                        'n_layers': depth,
                                        'd_model': width,
                                        'skip_type': 'residual',
                                        'paradigm': 'deterministic'
                                    }
                                    
                                    κ = self.compute_kappa(config)
                                    cost = self.compute_cost(config)
                                    
                                    # 제약 조건 확인
                                    if cost['flops'] > budget['flops']:
                                        continue
                                    if cost['memory'] > budget['memory']:
                                        continue
                                    
                                    # 목표 달성 가능?
                                    d_required = I_target / κ
                                    if d_required > width:
                                        continue
                                    
                                    # 효율 계산
                                    efficiency = κ / (cost['flops'] / 1e9)
                                    
                                    if efficiency > best_efficiency:
                                        best_efficiency = efficiency
                                        best_config = config
        
        return best_config, best_efficiency
    
    def tradeoff_analysis(self, base_config, alternatives):
        """선택지들의 트레이드오프 분석"""
        base_κ = self.compute_kappa(base_config)
        base_cost = self.compute_cost(base_config)
        
        results = []
        for alt_config in alternatives:
            alt_κ = self.compute_kappa(alt_config)
            alt_cost = self.compute_cost(alt_config)
            
            Δκ = (alt_κ - base_κ) / base_κ
            Δflops = (alt_cost['flops'] - base_cost['flops']) / base_cost['flops']
            Δmemory = (alt_cost['memory'] - base_cost['memory']) / base_cost['memory']
            
            results.append({
                'config': alt_config,
                'Δκ': Δκ,
                'Δflops': Δflops,
                'Δmemory': Δmemory,
                'efficiency': Δκ / (Δflops + 1e-9)
            })
        
        return sorted(results, key=lambda x: x['efficiency'], reverse=True)
```

---

### 4D. 실제 모델 검증

**목표:** 상수 기반 예측이 실제 모델에서 맞는지 검증

**테스트 모델:**
```
1. GPT-2 Small (124M params)
2. GPT-2 Medium (355M params)
3. ResNet-50
4. ResNet-101
5. ViT-Base
6. ViT-Large (축소 버전)
7. BERT-Base
8. LLaMA-style (7B 축소)
9. ConvNeXt-Tiny
10. Mamba (작은 버전)
```

**실험 설계:**
```python
def validate_model(model_name):
    # 1. 모델 구성 분석
    config = analyze_architecture(model_name)
    # {activation, norm, precision, attn_type, ffn_type, n_layers, d_model, ...}
    
    # 2. κ 예측
    κ_predicted = optimizer.compute_kappa(config)
    
    # 3. κ 실측
    model = load_pretrained(model_name)
    # 또는 scratch에서 학습
    κ_measured = compute_kappa_empirical(model, test_data)
    
    # 4. 오차 계산
    error = abs(κ_predicted - κ_measured) / κ_measured
    
    return {
        'model': model_name,
        'predicted': κ_predicted,
        'measured': κ_measured,
        'error': error
    }

results = []
for model in models:
    results.append(validate_model(model))

# 성공 기준: 평균 오차 < 10%
```

**예상 결과 테이블:**
```
| 모델 | 예측 κ | 실측 κ | 오차 |
|------|--------|--------|------|
| GPT-2 Small | 0.48 | 0.47 | 2.1% |
| GPT-2 Medium | 0.46 | 0.45 | 2.2% |
| ResNet-50 | 0.45 | 0.44 | 2.3% |
| ResNet-101 | 0.43 | 0.42 | 2.4% |
| ViT-Base | 0.47 | 0.46 | 2.2% |
| BERT-Base | 0.46 | 0.45 | 2.2% |
| LLaMA-style | 0.49 | 0.48 | 2.1% |
| ConvNeXt-T | 0.46 | 0.45 | 2.2% |
| Mamba | 0.50 | 0.49 | 2.0% |
```

**실험량:** 10종 × 3 seeds = 30 runs

---

### 4E. 설계 케이스 스터디

**Case 1: 모바일 최적화**
```
제약:
- FLOPs < 1G
- Memory < 100MB
- Latency < 10ms

ICON 기반 설계:
1. INT8 사용 (C_int8 = 0.92, 2x 속도)
2. SwiGLU 대신 ReLU (C_relu = 1.0, 적은 ops)
3. MQA 사용 (C_mqa = 1.03, 75% KV 절약)
4. 얕은 depth (6L, β^5 ≈ 0.99)

결과: κ_total ≈ 0.91, 제약 충족
```

**Case 2: 정확도 최대화**
```
제약:
- FLOPs 무제한
- Memory < 80GB

ICON 기반 설계:
1. FP32 유지 (C_fp32 = 1.0)
2. SwiGLU 사용 (C_swiglu = 1.02)
3. MHA 사용 (C_mha = 1.05)
4. 깊은 depth (32L)
5. Skip + LN

결과: κ_total ≈ 1.02, 최대 정보 용량
```

**Case 3: 효율성 밸런스**
```
제약:
- FLOPs < 10G
- Accuracy > baseline

ICON 분석:
- GELU → ReLU: κ -7%, FLOPs -10% → 효율 +3%
- FP32 → FP16: κ -2%, Memory -50% → 효율 +96%
- 12L → 16L: κ -0.8%, Params +33% → 효율 -34%

추천: FP16 + ReLU, 12L 유지
```

---

### 4F. 설계 도구 공개

**Python 라이브러리:**
```python
# icon_design/

icon_design/
├── __init__.py
├── constants.py      # ICON 상수 테이블
├── optimizer.py      # 최적화 알고리즘
├── analyzer.py       # 아키텍처 분석
├── predictor.py      # κ 예측
├── tradeoff.py       # 트레이드오프 분석
├── visualizer.py     # 시각화 도구
└── examples/
    ├── mobile_design.py
    ├── accuracy_max.py
    └── efficiency_balance.py
```

**사용 예시:**
```python
from icon_design import ICONDesigner

designer = ICONDesigner()

# 현재 모델 분석
current_κ = designer.predict_kappa({
    'activation': 'gelu',
    'normalization': 'layernorm',
    'precision': 'fp32',
    'attention_type': 'mha',
    'ffn_type': 'ffn_gelu',
    'n_layers': 12,
    'd_model': 768
})
print(f"Current κ: {current_κ}")

# 최적화
best_config = designer.optimize(
    I_target=1000,
    budget={'flops': 10e9, 'memory': 4e9}
)
print(f"Optimal config: {best_config}")

# 트레이드오프 분석
analysis = designer.tradeoff_analysis(
    current_config,
    alternatives=[
        {**current_config, 'precision': 'fp16'},
        {**current_config, 'activation': 'relu'},
        {**current_config, 'attention_type': 'gqa'},
    ]
)
designer.visualize_tradeoffs(analysis)
```

---

### 4G. Hardware Implications (Appendix)

**부록으로 이동 (검증 없이 가능성만 제시)**

```
# Appendix D: Hardware Design Implications

## 이론적 공식

1. 연산 유닛 배분
   N_ffn / N_attn = κ_attn / κ_ffn
   
   예: κ_attn = 1.05, κ_ffn = 0.96
   → FFN 유닛을 9% 더 할당

2. 메모리 대역폭
   M_required = d × L × (1/κ_layer) × precision_bytes
   
3. 전력 최적화
   P ∝ FLOPs × (1/κ_total)
   κ 높은 연산 = 같은 정보량에 더 적은 연산

## Future Work
- FPGA 검증
- ASIC 시뮬레이션
- 실제 칩 설계 협업
```

---

### 4편 총 실험량

| 섹션 | 실험 |
|------|------|
| 4D 모델 검증 | 30 runs |
| 4E 케이스 스터디 | 분석 (실험 아님) |
| **Total** | **~30 runs** |

---

### 4편 산출물

```yaml
# ICON-Design Guide v1.0

# 설계 공식
design:
  capacity_formula: "d = I_target / κ_total"
  kappa_formula: "κ = Π(C_i)"
  efficiency_formula: "E = Δκ / Δcost"

# 검증 결과
validation:
  gpt2_small: {predicted: 0.48, measured: 0.47, error: 2.1%}
  gpt2_medium: {predicted: 0.46, measured: 0.45, error: 2.2%}
  resnet50: {predicted: 0.45, measured: 0.44, error: 2.3%}
  vit_base: {predicted: 0.47, measured: 0.46, error: 2.2%}
  bert_base: {predicted: 0.46, measured: 0.45, error: 2.2%}
  average_error: 2.2%

# 설계 도구
toolkit:
  github: "https://github.com/xxx/icon-design"
  pip: "pip install icon-design"
```

---

### 4편 논문 구조

```
Abstract
1. Introduction
   - 1-3편 요약
   - 설계 응용의 필요성

2. Background
   - ICON 상수 recap
   - 기존 아키텍처 설계 방법

3. Design Methodology
   - 3.1 Capacity-Based Design
   - 3.2 Optimization Algorithm
   - 3.3 Tradeoff Analysis Framework

4. Validation
   - 4.1 Experimental Setup
   - 4.2 Real Model Predictions
   - 4.3 Error Analysis

5. Case Studies
   - 5.1 Mobile Optimization
   - 5.2 Accuracy Maximization
   - 5.3 Efficiency Balance

6. Design Toolkit
   - 6.1 Library Overview
   - 6.2 Usage Examples
   - 6.3 Extensibility

7. Discussion
   - 한계
   - 대규모 검증 필요성

8. Conclusion

Appendix A: 전체 검증 결과
Appendix B: 추가 케이스 스터디
Appendix C: 도구 문서
Appendix D: Hardware Implications (Future Work)
```

---
---

# 전체 요약

---

## 실험 총량

| 편 | 실험 |
|----|------|
| 1편 Primitive | ~153 runs |
| 2편 Block | ~141 runs |
| 3편 Paradigm | ~135 runs |
| 4편 Design | ~30 runs |
| **Total** | **~459 runs** |

---

## 시간 추정

**단일 실험 시간:** ~20-30분 (학습 + MI 측정)

**총 시간:**
```
459 runs × 25분 = 191시간 ≈ 8일

병렬화 (p4d 8 GPU):
191 / 8 = 24시간 ≈ 1일 per 편

실제 (코드 작성, 디버깅 포함):
~1주일 per 편
```

---

## 타임라인

```
Week 1-2: 1편 실험 + 논문
Week 3-4: 2편 실험 + 논문
Week 5-6: 3편 실험 + 논문
Week 7-8: 4편 실험 + 논문 + 도구

= 2개월 집중 → 4편 완성
```

---

## 리스크 관리

**Critical 검증 포인트:**
```
1편 1F: 독립성 검증 실패 시
→ 2차 상호작용 항 추가
→ "준독립적" 으로 재정의

1편 1G: 교차검증 실패 시
→ 조건별 상수 테이블 분리
→ "조건부 상수" 로 재정의

2편 2H: 블록 독립성 실패 시
→ 블록 조합 공식 수정
→ 경험적 보정 항 추가
```

---

## 논문 순서

```
1편: "기초가 상수다" → 프레임워크 확립
2편: "블록도 상수다" → 확장성 증명
3편: "패러다임도 상수다" → 일반성 증명
4편: "상수로 설계한다" → 실용성 증명
```

**각 편이 이전 편 위에 쌓이면서 독립적으로도 의미 있음.**

---

## 성공 기준

```
1편 성공 = 상수 개념 확립 + 독립성 증명
2편 성공 = 블록 수준 확장 + 조합 공식
3편 성공 = 패러다임 커버리지 + 교차검증
4편 성공 = 실제 모델 예측 정확도 < 10%

전체 성공 = "AI 정보 용량의 주기율표" 확립
```
