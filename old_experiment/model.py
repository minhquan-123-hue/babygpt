from tokenizer import encode, decode


class TinyModel:

    def __init__(self, vocab_size):
        # Các con số bên trong model
        self.weights = [
            [0.0 for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]

    def predict(self, current):
        # Lấy các điểm số của những từ có thể xuất hiện tiếp theo
        scores = self.weights[current]

        # Chọn từ có điểm cao nhất
        return max(range(len(scores)), key=lambda i: scores[i])

    def learn(self, current, answer):
        # Model tự dự đoán
        prediction = self.predict(current)

        # Nếu đoán sai
        if prediction != answer:

            # Giảm điểm của lựa chọn sai
            self.weights[current][prediction] -= 1

            # Tăng điểm của lựa chọn đúng
            self.weights[current][answer] += 1

        return prediction

