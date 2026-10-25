# Kết quả điều chuẩn hóa (15 epoch, seed [42], tập val)

| Run | n | Macro-F1 % (TB 3 ep cuối) | Acc % (TB 3 ep cuối) | Val loss cuối | Val-train loss | Train-val acc % | Recall lớp 23 % | Recall lớp 30 % | Recall lớp 0 % |
|---|---|---|---|---|---|---|---|---|---|
| A: baseline | 1 | 94.25 | 97.03 | 0.130 | 0.113 | 2.17 | 55.00 | 63.33 | 76.67 |
| B: dropout 0.5 | 1 | 97.27 | 98.43 | 0.059 | 0.017 | -0.04 | 83.33 | 80.00 | 80.00 |
| C: AdamW wd 1e-2 | 1 | 93.89 | 96.80 | 0.126 | 0.111 | 2.15 | 56.67 | 74.44 | 73.33 |
| D: label smoothing 0.1 | 1 | 97.08 | 98.07 | 0.232 | 0.113 | 2.32 | 86.67 | 75.56 | 80.00 |
| E: B+C+D | 1 | 97.94 | 98.78 | 0.202 | 0.015 | 0.84 | 84.17 | 84.44 | 80.00 |

## So với baseline (macro-F1, TB 3 epoch cuối)
- B: dropout 0.5: +3.02% -> tốt hơn
- C: AdamW wd 1e-2: -0.36% -> trong ngưỡng nhiễu, chưa kết luận được
- D: label smoothing 0.1: +2.83% -> tốt hơn
- E: B+C+D: +3.69% -> tốt hơn

Chú thích: loss ghi log là cross-entropy thường (không smoothing) nên so được giữa các run. Val-train loss và Train-val acc càng nhỏ thì overfit càng ít. Recall từng lớp lấy từ checkpoint tốt nhất (val acc cao nhất), còn các cột còn lại tính ở cuối huấn luyện.