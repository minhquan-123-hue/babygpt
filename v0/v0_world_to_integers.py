"""
BabyGPT - Version 0 (Siêu đơn giản)

Mục tiêu duy nhất của file này:
Chứng minh rằng text, image, audio và chuyển động vật lý
đều có thể được biến thành dãy số nguyên (list of integers).

Không dùng bất kỳ thư viện nào ngoài Python chuẩn.
Không train, không neural network, không generate phức tạp.
Chỉ là bước "nhìn thế giới → thành số".
"""

# ============================================================
# 1. TEXT → dãy số nguyên
# ============================================================
# Ý tưởng: mỗi từ là một số.
# Đây là tokenizer đơn giản nhất có thể.

WORD_TO_ID = {
    "tôi": 0,
    "ăn": 1,
    "cơm": 2,
    "uống": 3,
    "nước": 4,
    "mèo": 5,
    "cá": 6,
    "chó": 7,
    "ngủ": 8,
    "đi": 9,
    "chạy": 10,
    "nhảy": 11,
}

ID_TO_WORD = {v: k for k, v in WORD_TO_ID.items()}


def text_to_integers(text: str) -> list[int]:
    """
    Biến câu tiếng Việt thành dãy số nguyên.
    Ví dụ: "tôi ăn cơm" → [0, 1, 2]
    """
    words = text.lower().strip().split()
    result = []
    for word in words:
        # Nếu từ chưa có trong từ điển thì bỏ qua (hoặc có thể gán số đặc biệt)
        if word in WORD_TO_ID:
            result.append(WORD_TO_ID[word])
    return result


def integers_to_text(ids: list[int]) -> str:
    """Ngược lại: dãy số → câu."""
    return " ".join(ID_TO_WORD.get(i, "???") for i in ids)


# ============================================================
# 2. IMAGE → dãy số nguyên
# ============================================================
# Ý tưởng cực đơn giản:
# Một ảnh 4x4 grayscale (chỉ để minh họa).
# Mỗi pixel là một số từ 0 → 255.
# Ta "xếp phẳng" ảnh thành 1 dãy số dài.

def make_tiny_image() -> list[list[int]]:
    """
    Tạo một ảnh giả 4x4 (giả lập).
    Trong thực tế sau này sẽ đọc file ảnh thật.
    """
    # Ảnh đơn giản: một "ô vuông sáng" ở giữa
    return [
        [10, 10, 10, 10],
        [10, 200, 200, 10],
        [10, 200, 200, 10],
        [10, 10, 10, 10],
    ]


def image_to_integers(image: list[list[int]]) -> list[int]:
    """
    Biến ảnh 2D thành dãy số 1D.
    Ví dụ 4x4 → list 16 số.
    """
    result = []
    for row in image:
        for pixel in row:
            result.append(pixel)
    return result


def integers_to_image(ids: list[int], width: int = 4) -> list[list[int]]:
    """Ngược lại: dãy số → ảnh 2D."""
    image = []
    for i in range(0, len(ids), width):
        image.append(ids[i : i + width])
    return image


# ============================================================
# 3. AUDIO → dãy số nguyên
# ============================================================
# Ý tưởng cực đơn giản:
# Âm thanh là sóng. Ta lấy vài mẫu biên độ (amplitude).
# Mỗi mẫu là một số nguyên (ví dụ từ -100 → 100).

def make_tiny_audio() -> list[int]:
    """
    Tạo một đoạn "âm thanh" giả rất ngắn.
    Trong thực tế sau này sẽ đọc file .wav.
    Đây chỉ là ví dụ minh họa: sóng hình sin đơn giản.
    """
    # Biên độ giả lập (không dùng math.sin để giữ thuần Python)
    return [0, 30, 50, 30, 0, -30, -50, -30, 0, 20, 40, 20, 0]


def audio_to_integers(samples: list[int]) -> list[int]:
    """
    Âm thanh đã là dãy số rồi.
    Hàm này chỉ để giữ tên gọi thống nhất.
    """
    return samples[:]  # copy cho chắc


# ============================================================
# 4. PHYSICAL MOTION (chuyển động) → dãy số nguyên
# ============================================================
# Ý tưởng:
# Một vật di chuyển trong không gian 2D.
# Mỗi bước thời gian ta ghi lại (x, y).
# Ta có thể gộp thành dãy số: [x1, y1, x2, y2, ...]

def make_tiny_motion() -> list[tuple[int, int]]:
    """
    Tạo quỹ đạo chuyển động giả của một điểm.
    Ví dụ: đi từ (0,0) → (3,0) → (3,2)
    """
    return [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (3, 2),
    ]


def motion_to_integers(path: list[tuple[int, int]]) -> list[int]:
    """
    Biến danh sách tọa độ thành dãy số phẳng.
    Ví dụ: [(0,0), (1,0)] → [0, 0, 1, 0]
    """
    result = []
    for x, y in path:
        result.append(x)
        result.append(y)
    return result


def integers_to_motion(ids: list[int]) -> list[tuple[int, int]]:
    """Ngược lại: dãy số → danh sách tọa độ."""
    path = []
    for i in range(0, len(ids), 2):
        if i + 1 < len(ids):
            path.append((ids[i], ids[i + 1]))
    return path


# ============================================================
# 5. DEMO: Chạy thử toàn bộ
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("BABYGPT VERSION 0 - Thế giới → Dãy số nguyên")
    print("=" * 50)

    # --- TEXT ---
    text = "tôi ăn cơm"
    text_ids = text_to_integers(text)
    print("\n[TEXT]")
    print("  Gốc      :", text)
    print("  → Integers:", text_ids)
    print("  ← Decode  :", integers_to_text(text_ids))

    # --- IMAGE ---
    image = make_tiny_image()
    image_ids = image_to_integers(image)
    print("\n[IMAGE 4x4]")
    print("  Gốc (ma trận):")
    for row in image:
        print("   ", row)
    print("  → Integers:", image_ids)
    print("  ← Decode  :")
    for row in integers_to_image(image_ids):
        print("   ", row)

    # --- AUDIO ---
    audio = make_tiny_audio()
    audio_ids = audio_to_integers(audio)
    print("\n[AUDIO]")
    print("  Gốc (biên độ):", audio)
    print("  → Integers   :", audio_ids)

    # --- MOTION ---
    motion = make_tiny_motion()
    motion_ids = motion_to_integers(motion)
    print("\n[PHYSICAL MOTION]")
    print("  Gốc (tọa độ):", motion)
    print("  → Integers  :", motion_ids)
    print("  ← Decode    :", integers_to_motion(motion_ids))

    print("\n" + "=" * 50)
    print("Kết luận Version 0:")
    print("  Mọi thứ (text / image / audio / motion)")
    print("  đều đã được biến thành dãy số nguyên.")
    print("  Đây chính là nền tảng để sau này build model.")
    print("=" * 50)
