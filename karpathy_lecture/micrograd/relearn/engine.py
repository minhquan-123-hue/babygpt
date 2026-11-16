import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self.op = _op
        self.backward = lambda: None
        self.children = set(_children)

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self,other):
        new_obj = Value(self.data + other.data, (self,other))

        def backward():
            self.grad += 1.0 * new_obj.grad
            other.grad += 1.0 * new_obj.grad

        new_obj.backward = backward

        return new_obj

    def __mul__(self,other):
        new_obj = Value(self.data * other.data, (self,other))

        def backward():
            self.grad = other.data * new_obj.grad
            other.grad = self.data * new_obj.grad

        new_obj.backward = backward
        return new_obj

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def backward():
            self.grad += (1 - out.data**2) * out.grad

        out.backward = backward
        return out


### dừng copy lại vì đã nhớ lại phần ý tưởng
### đầu tiên là tính toán
### tất các các phép tính forward pass
### sau đó khi có prediction thì so sánh nó với phần
### dữ liệu mong muốn , tính toán sai số
### lan truyền ngược độ ảnh hưởng cho tất cả các node
### sau đó học dựa trên grad info if posi giảm if nega tăng
### sau đó tính lại forward pass 
### "đo" lại sai số lần này là bao nhiêu
### và lặp đi lặp lại quá trình đến khi prediction match với 
### dữ liệu ta mong muốn

### phần đúc kết là weight sau khi học với nhiều input
### sẽ giữ nguyên và dù ném inputs nào vào nó cũng
### sẽ tạo ra output mong muốn
### có thể đọc phần giải thích lại ở micrograd/bigdata.py 

### so sánh sự khác biệt giữa kiểu tính loss()
### tính loss theo kiểu mô hình hiện tại là chỉ 1 input là int , và (pre - out)^2 
### tính loss theo mô hình biagram , sẽ dùng log(likelihood) 
### likelihood là xác suất của ký tự có khả năng xảy ra tiếp
### theo chưa phải là sai số , vậy cần phải sử dụng một cách 
### nào đó , hàm số nào đó , các nhà sci&rese đã nghĩ ra 
### dùng log(likelihood) để biểu diễn sai số và hiệu quả thì dùng thôi , stop asking non-stop 
### và dữ liệu của loss ta muốn biểu diễn nó dương nên ta
### đảo chiều log(likelihood) thành negative log(likelihood)
### kết thúc giải thích , ta đã hiểu là nếu mà nll + spring_force thì loss sẽ được cộng thêm 1 khoảng nhất định
### và nó sẽ làm giảm thêm weight xuống