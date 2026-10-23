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













## Mô hình "tự tin" bao nhiêu thì biết bằng cách nào

Với mỗi ảnh, mô hình trả về 43 con số thô (logits). Hàm `predict` đưa chúng qua `softmax` để thành 43 xác suất, cộng lại bằng 1:

```python
probs = torch.softmax(model(x), dim=1)
conf, pred = probs.max(dim=1)
```

- `pred` là lớp có xác suất cao nhất, tức đáp án mô hình chọn.
- `conf` là **xác suất của chính lớp đó**, và đây là con số trong ngoặc bên cạnh ảnh sai.

Ví dụ ảnh có `(1.00)` nghĩa là mô hình gán xấp xỉ 100% xác suất cho đáp án sai của nó. Ảnh có `(0.45)` nghĩa là mô hình còn lưỡng lự, lớp đoán chỉ hơn các lớp khác chút ít. Số hiển thị đã làm tròn 2 chữ số, nên `1.00` có thể là 0.996 hay 0.9999.

Con số này **không phải** xác suất của lớp thật, cũng không phải accuracy. Nó chỉ cho biết mô hình chắc chắn đến đâu về đáp án (sai) của mình.

## Vì sao biết ảnh nào tự tin nhất

Chỉ cần sắp xếp các ảnh sai theo `conf` giảm dần rồi lấy `n` ảnh đầu:

```python
wrong = np.where(labels != preds)[0]      # chỉ số các ảnh bị đoán sai
wrong = wrong[np.argsort(confs[wrong])[::-1]][:n]
```

- `np.where(labels != preds)[0]`: lấy vị trí của các ảnh mà nhãn thật khác nhãn dự đoán.
- `confs[wrong]`: lấy độ tự tin của riêng những ảnh sai đó.
- `np.argsort(...)`: trả về thứ tự vị trí từ `conf` thấp đến cao.
- `[::-1]`: đảo ngược thành cao đến thấp.
- `[:n]`: giữ lại 12 ảnh đầu.

## Vì sao chọn ảnh "sai mà tự tin nhất"

Đó là các ảnh mang nhiều thông tin nhất. Ảnh sai với conf thấp thường là ảnh mờ hoặc khó, mô hình đã do dự và lỗi đó dễ hiểu. Ảnh sai với conf rất cao cho thấy một trong ba khả năng:
- Hai lớp thực sự trông gần như giống nhau (ví dụ hai biển giới hạn tốc độ).
- Mô hình học nhầm một đặc điểm không đúng.
- **Nhãn trong dữ liệu có thể sai.** Nếu nhìn ảnh mà thấy mô hình có vẻ đúng hơn nhãn thật, đó là một phát hiện đáng ghi vào báo cáo.

Một lưu ý nữa: mạng nơ-ron thường **quá tự tin**, nên conf = 1.00 không có nghĩa là xác suất sai thật sự bằng 0. Hãy dùng nó để xếp thứ tự các ảnh cần xem lại, đừng xem nó như xác suất chính xác. Nếu muốn xem cả các ảnh sai với mức tự tin trung bình, bạn có thể đổi `[::-1]` thành lấy ngẫu nhiên, hoặc tăng `n` để thấy nhiều ảnh hơn.