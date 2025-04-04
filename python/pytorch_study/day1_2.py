import torch as t
import torch.nn as nn

x = t.randn(4,3)
y = t.randn(4,2)

linear = nn.Linear(3,2)

print('w: ', linear.weight)
print('b: ', linear.bias)

criterion = nn.MSELoss()
optimizer = t.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss: ', loss.item())

loss.backward()

print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)


optimizer.step()

pred = linear(x)
loss = criterion(pred, y)

print('loss after 1 step optimization', loss.item())