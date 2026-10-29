from micrograd_from_scratch import Value

def one_neuron():
    x1 = Value(2,label = 'x1')
    w1 = Value(3,label = 'w1')
    x2 = Value(4,label = 'x2')
    w2 = Value(5,label = 'w2')
    b = Value(1,label = 'b')
    
    x1w1 = x1 * w1; x1w1._label = 'x1w1'
    x2w2 = x2 * w2; x2w2._label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2._label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n._label = 'n'
    o = n.tanh();o._label = 'o'
    
    
    o.backward()

def main():
    a = Value(0.0, label = 'a')
    n = a + 1.0
    print(n)


if __name__ == "__main__":
    main() 