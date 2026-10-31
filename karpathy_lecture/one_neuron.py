from micrograd import Value

def one_neuron():
    x1 = Value(2.0,_label = 'x1')
    w1 = Value(0.0,_label = 'w1')
    x2 = Value(-3.0,_label = 'x2')
    w2 = Value(1.0,_label = 'w2')
    b = Value(6.8813735970,_label = 'b')
    
    x1w1 = x1 * w1; x1w1._label = 'x1w1'
    x2w2 = x2 * w2; x2w2._label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2._label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n._label = 'n'
    o = n.tanh();o._label = 'o'
    
    
    o.backward()
    print(x1)
    print("*")
    print(w1)
    print("=")
    print(x1w1)
    print()
    print()
    print(x2)
    print("*")
    print(w2)
    print("=")
    print(x2w2)
    print()
    print()
    print(x1w1)
    print("+")
    print(x2w2)
    print("=")
    print(x1w1x2w2)
    print()
    print()
    print(x1w1x2w2)
    print("+")
    print(b)
    print("=")
    print(n)
    print("=>")
    print(o)



def test():
    x1 = Value(2.0,_label = 'x1')
    w1 = Value(0.0,_label = 'w1')
    x2 = Value(-3.0,_label = 'x2')
    w2 = Value(1.0,_label = 'w2')
    b = Value(6.8813735970,_label = 'b')
    
    x1w1 = x1 * w1; x1w1._label = 'x1w1'
    x2w2 = x2 * w2; x2w2._label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2._label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n._label = 'n'

    
    e = 2**(2 * n)
    o = (e + 1) / (e - 1)

    o.backward()


def main():
    a = Value(1.0)
    o = a**1/2
    print(o)

if __name__ == "__main__":
    main() 