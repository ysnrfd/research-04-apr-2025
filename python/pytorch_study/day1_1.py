import torch as t

import torch.nn as nn

import numpy as np

x = t.tensor(1., requires_grad=True)
w = t.tensor(2., requires_grad=True)
b = t.tensor(3., requires_grad=True)

y = w*x+b

y.backward()



print(x.grad)
print(w.grad)
print(b.grad)