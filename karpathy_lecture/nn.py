from micrograd import Value
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        act = sum(wi*xi for wi,xi in zip(self.w , x)) + self.b
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self,nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # day la mot list cua int
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

    


n = MLP(3,[4,4,1])

xs = [
    (2.0, 3.0, -1.0),
    (3.0, -1.0, 0.5),
    (0.5, 1.0, 1.0),
    (1.0, 1.0, -1.0)
]

ys = [1.0, -1.0, -1.0, 1.0] # desired targets
ypred = [n(x) for x in xs]

print(ypred)

print()
loss = sum([(yout - ygt)**2 for yout, ygt in zip(ypred, ys)])
print(loss)

loss.backward()

print(len(n.parameters()))

print(n.layers[0].neurons[0].w[0].grad)
