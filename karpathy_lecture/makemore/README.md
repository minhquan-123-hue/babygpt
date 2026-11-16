Neural Network: Prediction, Loss & Learning
1. Forward Pass và Probability

Neural Network nhận Input, đưa qua Linear Layer để tạo ra Logits. Sau đó, sử dụng Exponential để chuyển Logits thành các giá trị dương chưa chuẩn hóa (fake counts), rồi sử dụng Softmax để chuẩn hóa thành Probability Distribution có tổng bằng 1.

2. Loss Calculation

Probability chưa phải Loss. Chúng ta sử dụng Target để xác định xác suất tương ứng với kết quả thực tế, rồi tính Negative Log-Likelihood:

Loss=-ln(P)


Sau đó cộng tất cả Loss và chia cho số lần dự đoán để tính Average Negative Log-Likelihood.

3. Backpropagation

Khi có Average Loss, khởi tạo gradient của nó bằng 1.0. Sử dụng Chain Rule và local derivatives để lan truyền gradient ngược qua toàn bộ Computational Graph, từ Total Loss đến các node trước đó, bao gồm cả Leaf Nodes.

Lưu ý: Nếu Loss là trung bình của N quan sát, gradient truyền đến mỗi Loss thành phần sẽ được nhân với (1/N).

4. Parameter Update

Sau khi tính được gradient, chúng ta xác định mức độ ảnh hưởng của từng Trainable Parameter lên Average Loss.

Mô hình hiện tại sử dụng Linear Layer không có Bias, nên chỉ cập nhật Weights bằng Gradient Descent:

w_new += -0.01 * grad


5. Learning

Sau khi Weights thay đổi, mô hình thực hiện Forward Pass mới. Logits và Probability Distribution cũng thay đổi theo.

Quá trình này lặp lại nhiều lần để giảm Average Negative Log-Likelihood, giúp mô hình ngày càng phân bổ xác suất tốt hơn cho những ký tự thực tế xuất hiện trong Dataset.