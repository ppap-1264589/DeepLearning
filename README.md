# Vấn đề chia dữ liệu từ tập train thành hai tập: train và val

Dữ liệu train được chia thành train/val (80/20) bằng StratifiedGroupKFold: các frame cùng một track luôn nằm chung một tập để tránh rò rỉ, và phân bố lớp được giữ xấp xỉ giữa hai tập. 

Tập test giữ nguyên. 

Mỗi ảnh được đọc bởi GTSRBDataset, crop theo ROI, rồi qua một bộ biến đổi: tập train dùng bộ có augmentation ngẫu nhiên (xoay, dịch chuyển, đổi độ sáng), tập val/test chỉ resize 48×48 và chuẩn hóa. DataLoader gom các ảnh đã xử lý thành batch. 

Để xử lý mất cân bằng lớp, tập train có thể dùng WeightedRandomSampler với trọng số tỉ lệ nghịch số ảnh của lớp, giúp các lớp được lấy mẫu với xác suất xấp xỉ bằng nhau; 

val/test không dùng sampler để phản ánh phân bố thật.


# Vấn đề đặt giá trị SEED cố định

Nếu không đặt seed, mỗi lần chạy sẽ ra kết quả hơi khác nhau dù code và cấu hình giống hệt. Có seed thì bạn chạy lại sẽ thấy gần như đúng các con số cũ.

Vì sao điều này quan trọng với project của bạn:

Khi so sánh Adam với SGD, hoặc có/không scheduler, bạn muốn chênh lệch kết quả đến từ cấu hình, không phải do may rủi lúc khởi tạo hoặc thứ tự dữ liệu. Cùng một seed nghĩa là các run bắt đầu từ cùng trọng số ban đầu, nên so sánh công bằng hơn. Nó cũng giúp bạn và nhóm tái hiện được kết quả đã đưa vào báo cáo.






## Chỉ số đánh giá: F1 và Macro-F1

Bài toán có 43 lớp biển báo với số ảnh mỗi lớp chênh lệch nhau, nên ngoài Accuracy chúng tôi dùng thêm **Macro-F1**.

### Precision, Recall và F1

Với mỗi lớp (ví dụ "Giới hạn tốc độ 30"):

- **Precision**: trong số ảnh mô hình dự đoán là lớp này, bao nhiêu ảnh thật sự thuộc lớp này. Precision thấp nghĩa là mô hình hay báo nhầm.
- **Recall**: trong số ảnh thật sự thuộc lớp này, mô hình tìm ra được bao nhiêu. Recall thấp nghĩa là mô hình hay bỏ sót.
- **F1** là trung bình điều hòa của hai chỉ số trên:

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1 chỉ cao khi **cả** Precision và Recall đều cao. Một lớp có Precision = 1.0 nhưng Recall = 0.1 chỉ đạt F1 ≈ 0.18, trong khi trung bình cộng cho 0.55 và che mất điểm yếu.

### Macro-F1

Macro-F1 là trung bình **đều** của F1 trên 43 lớp, mỗi lớp có trọng số như nhau dù có nhiều hay ít ảnh:

$$\text{Macro-F1} = \frac{1}{43} \sum_{c=1}^{43} F1_c$$

Vì vậy, nếu một vài lớp ít ảnh bị đoán kém, Macro-F1 giảm rõ rệt trong khi Accuracy gần như không đổi. Khi Accuracy cao nhưng Macro-F1 thấp hơn đáng kể, đó là dấu hiệu có lớp đang bị bỏ rơi.

### Cách dùng trong dự án

| Chỉ số | Tập dữ liệu | Thời điểm tính |
|---|---|---|
| Loss, Accuracy | train và val | Mỗi epoch trong lúc huấn luyện (`train.py`) |
| Macro-F1 | train và val | Mỗi epoch, biểu đồ chỉ vẽ đường val |
| Accuracy, Macro-F1, F1 theo lớp, confusion matrix | test | Một lần duy nhất ở cuối, sau khi đã chốt mô hình |

- Tập **val** dùng để theo dõi quá trình học, phát hiện overfitting, chọn checkpoint và so sánh các cấu hình (Adam với SGD, có hoặc không scheduler, ...).
- Tập **test** chỉ dùng để báo cáo kết quả cuối, không dùng để chọn mô hình.
- F1 theo từng lớp và confusion matrix cho biết lỗi nằm ở Precision (hay gán nhầm vào lớp đó) hay Recall (hay bỏ sót lớp đó). Với biển báo giao thông, bỏ sót một biển như STOP (Recall thấp) thường nguy hiểm hơn việc báo nhầm.