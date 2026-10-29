from micrograd_from_scratch import Value

#khi ma input duoc dung nhieu lan backpropagation phai lan truyen nguoc lai
#va no phai duoc cong vao them va grad cua input do
a = Value(1.2 , label = 'a')
b = a + a; b._label = 'b'
b.backward()
print(a)
print(b)