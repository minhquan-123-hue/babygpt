def f(x):
    return 3*x**2 - 4*x + 5

print(f(3.0), "y lúc đầu")
print(f(3.0 + 0.001), "y lúc sau")
print((f(3.0 + 0.001) - f(3.0)) / 0.001, "đây là tỷ lệ 2 cạnh góc vuông")