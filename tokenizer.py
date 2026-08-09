# tokenizer.py

# để làm việc với con số nguyên
word_to_id = {
    "tôi": 0,
    "ăn": 1,
    "cơm": 2,
    "uống": 3,
    "nước": 4,
    "mèo": 5,
    "cá": 6,
    "chó": 7,
    "ngủ": 8
}

id_to_word = {
    0: "tôi",
    1: "ăn",
    2: "cơm",
    3: "uống",
    4: "nước",
    5: "mèo",
    6: "cá",
    7: "chó",
    8: "ngủ"
}


def encode(text):
    # biến thành list của các ký tự 
    words = text.split()
    # truy cập vào key và điền nó vào một list
    # chứa toàn bộ từ điện hiện có
    return [word_to_id[word] for word in words]


def decode(ids):
    return " ".join(id_to_word[i] for i in ids)


