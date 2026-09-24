# BabyGPT – Version 0 (Siêu đơn giản)

## Mục tiêu của version này

Chỉ làm **một việc duy nhất**:

> Chứng minh rằng text, image, audio và chuyển động vật lý  
> đều có thể được biến thành **dãy số nguyên (list of integers)**.

Không train model.  
Không dùng PyTorch / numpy / bất kỳ thư viện nào.  
Không generate ảnh đẹp hay âm thanh hay.  

Chỉ là bước đầu tiên: **nhìn thế giới → thành số**.

---

## Cách chạy

```bash
python v0_world_to_integers.py
```

Bạn sẽ thấy 4 phần:

1. Text → integers  
2. Image (4×4 giả) → integers  
3. Audio (biên độ giả) → integers  
4. Physical motion (tọa độ) → integers  

---

## Ý tưởng cốt lõi cần hiểu

| Thế giới thực          | Cách biến thành số nguyên                  | Ví dụ |
|------------------------|--------------------------------------------|-------|
| Câu chữ                | Mỗi từ → 1 số                              | "tôi ăn cơm" → [0, 1, 2] |
| Ảnh                    | Mỗi pixel → 1 số, xếp phẳng                | 4×4 → list 16 số |
| Âm thanh               | Mỗi mẫu biên độ → 1 số                     | [0, 30, 50, ...] |
| Chuyển động vật lý     | Mỗi bước (x, y) → 2 số                     | [(0,0),(1,0)] → [0,0,1,0] |

Sau khi đã có dãy số, **mọi model sau này** (dù đơn giản hay phức tạp)  
chỉ cần học quan hệ giữa các dãy số đó thôi.

---

## Lộ trình tiếp theo (sẽ build dần)

- **v0** (file này): Thế giới → integers  
- **v1**: Model cực đơn giản (chỉ bảng đếm / bigram) học từ text integers  
- **v2**: Thêm image integers vào cùng không gian  
- **v3**: Thêm audio + motion  
- **v4**: Bắt đầu có neural network thật (nhưng vẫn giữ code rõ ràng)  
- ... dần dần lên 32×32, 3 giây audio, 200 text

Không nhảy cóc. Mỗi version chỉ thêm **một** ý tưởng mới.
