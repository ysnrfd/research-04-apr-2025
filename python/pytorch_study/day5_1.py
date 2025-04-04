import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class neural_network (nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(neural_network, self).__init__()
        self.hidden = nn.Linear (input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.output = nn.Linear (hidden_dim, output_dim)

    def forward (self, x):
        x = self.hidden (x)
        x = self.act (x)
        x = self.output (x)
        return x


input_dim = 4
hidden_dim = 32
output_dim = 4

model = neural_network(input_dim, hidden_dim, output_dim)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.01)