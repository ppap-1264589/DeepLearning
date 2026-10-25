# Kết quả điều chuẩn hóa (15 epoch, seed [42, 43, 44], tập val)

| Run | n | Macro-F1 % (TB 3 ep cuối) | Acc % (TB 3 ep cuối) | Val loss cuối | Val-train loss | Train-val acc % | Recall lớp 23 % | Recall lớp 30 % | Recall lớp 0 % |
|---|---|---|---|---|---|---|---|---|---|
| A: baseline | 3 | 94.96 ± 0.64 | 97.27 ± 0.22 | 0.120 ± 0.010 | 0.104 ± 0.010 | 2.07 ± 0.14 | 60.83 ± 6.29 | 75.93 ± 10.91 | 78.89 ± 3.85 |
| B: dropout 0.5 | 3 | 97.08 ± 0.70 | 98.32 ± 0.44 | 0.074 ± 0.027 | 0.036 ± 0.026 | 0.48 ± 0.51 | 87.50 ± 11.02 | 76.67 ± 4.84 | 84.44 ± 7.70 |
| C: AdamW wd 1e-2 | 3 | 94.92 ± 0.94 | 97.22 ± 0.47 | 0.142 ± 0.036 | 0.129 ± 0.037 | 2.21 ± 0.56 | 56.67 ± 9.17 | 71.11 ± 3.33 | 77.78 ± 5.09 |
| D: label smoothing 0.1 | 3 | 97.30 ± 0.55 | 98.34 ± 0.32 | 0.217 ± 0.014 | 0.099 ± 0.013 | 1.71 ± 0.56 | 88.61 ± 10.55 | 74.81 ± 0.64 | 78.89 ± 1.92 |
| E: B+C+D | 3 | 98.18 ± 0.22 | 98.90 ± 0.10 | 0.179 ± 0.035 | -0.006 ± 0.033 | 0.78 ± 0.08 | 92.78 ± 8.01 | 82.22 ± 2.94 | 77.78 ± 1.92 |

## So với baseline (macro-F1, TB 3 epoch cuối)
- B: dropout 0.5: +2.11% -> tốt hơn
- C: AdamW wd 1e-2: -0.04% -> trong ngưỡng nhiễu, chưa kết luận được
- D: label smoothing 0.1: +2.34% -> tốt hơn
- E: B+C+D: +3.22% -> tốt hơn

Chú thích: loss ghi log là cross-entropy thường (không smoothing) nên so được giữa các run. Val-train loss và Train-val acc càng nhỏ thì overfit càng ít. Recall từng lớp lấy từ checkpoint tốt nhất (val acc cao nhất), còn các cột còn lại tính ở cuối huấn luyện.