# Model Implementation

| Component | Implementation |
| --- | --- |
| Area and cloud embedding | `STVGLGFormer.area_embedding` |
| Calendar embedding | `STVGLGFormer.calendar_embedding` |
| Regional environmental embedding | `STVGLGFormer.env_embedding` |
| Variable, region, and temporal attention | `AxialEnvironmentBlock` |
| Area-centered cross-attention | `STVGLGFormer.time_cross`, `STVGLGFormer.env_cross` |
| Hierarchical compatibility gates | `STVGLGFormer.gate_time`, `STVGLGFormer.gate_env` |
| Three-route feature fusion | `STVGLGFormer.route_logits` |
| Lifecycle attention pooling | `softmax(tanh(Z), dim=time)` |
| Historical condition | `STVGLGFormer.condition` |
| Future-date queries | `STVGLGFormer.query_blocks` |
| Ordered lifecycle coordinates | `STVGLGFormer.warp_head` |
| Shared GLG parameters | `STVGLGFormer.parameter_head`, `glg_parameters` |
| Gaussian, Logistic, and Gompertz bases | `glg_basis` |
| Area fitting loss | `masked_mse` |
| Progressive-prefix residual objective | `progressive_objective`, `preceding_batch` |

| Parameter | Value |
| --- | --- |
| Hidden width | 128 |
| Attention heads | 8 |
| STV encoder blocks | 4 |
| Future-query blocks | 2 |
| SwiGLU hidden width | 512 |
| Maximum sequence length | 177 |
| Minimum historical prefix | 7 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 4 |
| Updates per epoch | 30 |
| Maximum epochs | 60 |
| Gradient norm limit | 1.0 |
| Progressive loss weight | 0.2 |
| Progressive tolerance | 25 km4 |
| Random seed | 42 |

Training settings are defined in `configs/`. Prediction areas are returned in km2.
