Dữ liệu train được chia thành train/val (80/20) bằng StratifiedGroupKFold: các frame cùng một track luôn nằm chung một tập để tránh rò rỉ, và phân bố lớp được giữ xấp xỉ giữa hai tập. 

Tập test giữ nguyên. 

Mỗi ảnh được đọc bởi GTSRBDataset, crop theo ROI, rồi qua một bộ biến đổi: tập train dùng bộ có augmentation ngẫu nhiên (xoay, dịch chuyển, đổi độ sáng), tập val/test chỉ resize 48×48 và chuẩn hóa. DataLoader gom các ảnh đã xử lý thành batch. 

Để xử lý mất cân bằng lớp, tập train có thể dùng WeightedRandomSampler với trọng số tỉ lệ nghịch số ảnh của lớp, giúp các lớp được lấy mẫu với xác suất xấp xỉ bằng nhau; 

val/test không dùng sampler để phản ánh phân bố thật.