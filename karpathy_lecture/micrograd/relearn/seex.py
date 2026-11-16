a = 2.0
b = -3.0
c = 10.0

# pump inputs
h = 0.001

# result before bump
d1 = a*b +c
print(d1)

#a += h , bump a
#b += h , bump b
c += h

# result after bump
d2 = a*b +c
print(d2)
slope = (d2 - d1) / h
print(slope)