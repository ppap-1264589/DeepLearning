# Kết quả lưới thí nghiệm

15 epoch, seed [42], đo trên tập val.

| Optimizer | Scheduler | Balance | n | Best ep | Val acc % | Val macro-F1 % | Acc cuối % | F1 cuối % | Val loss cuối | Gap train-val % |
|---|---|---|---|---|---|---|---|---|---|---|
| adam | plateau | none | 1 | 8 | 97.70 | 96.53 | 97.16 | 94.96 | 0.206 | 2.66 |
| adam | none | none | 1 | 12 | 97.60 | 96.25 | 97.35 | 95.40 | 0.130 | 2.17 |
| adam | plateau | sampler | 1 | 15 | 97.24 | 95.08 | 97.24 | 95.08 | 0.156 | 2.62 |
| sgd | none | none | 1 | 11 | 96.77 | 94.80 | 96.27 | 93.15 | 0.213 | 3.08 |
| adam | cosine | none | 1 | 13 | 97.20 | 94.79 | 97.20 | 94.67 | 0.165 | 2.68 |
| sgd | plateau | none | 1 | 13 | 96.51 | 93.43 | 96.34 | 93.09 | 0.219 | 3.31 |
| sgd | cosine | none | 1 | 13 | 96.17 | 93.03 | 95.99 | 92.75 | 0.212 | 3.59 |

Best ep = epoch có val acc cao nhất (checkpoint đã lưu). Acc cuối / F1 cuối = giá trị ở epoch cuối, so với Best để thấy cấu hình nào ổn định. Gap = train acc - val acc ở epoch cuối.

# Quá trình chọn

## Bước 1: lưới optimizer x scheduler (balance=none)
Cùng 15 epoch; Adam lr 1e-3, SGD lr 1e-2. Xếp theo val macro-F1 tại epoch có val acc cao nhất (chính là checkpoint đã lưu).
- Hạng 1: adam+plateau: F1 96.53%, acc 97.70%
- Hạng 2: adam+none: F1 96.25%, acc 97.60%
- Hạng 3: sgd+none: F1 94.80%, acc 96.77%

## Bước 2: bỏ qua (chỉ 1 seed, dùng --seeds 42 43 44 để giảm nhiễu)
- Cảnh báo: adam+plateau hơn adam+none chỉ 0.28% F1, nằm trong ngưỡng nhiễu 1% của val. Chọn theo thứ hạng nhưng chưa nên coi là khác biệt thật.
=> Chọn: **adam + plateau** (lr 0.001)

## Bước 3: WeightedRandomSampler (A/B trên cấu hình đã chọn)
- none   : F1 96.53%, acc 97.70%
- sampler: F1 95.08%, acc 97.24%
- Chênh lệch (sampler - none): F1 -1.45%, acc -0.46%
=> Sampler làm macro-F1 kém đi rõ rệt: không dùng.