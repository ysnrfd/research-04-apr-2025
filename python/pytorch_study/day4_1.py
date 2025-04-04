import torch
import torch.nn as nn
import torch.nn.functional as F
class simple_sd_model (nn.Module):
    def __init__(self):
        super(simple_sd_model, self).__init__()
        self.fc1 = nn.Linear (1,1)
        self.fc2 = nn.Linear (1,6)
        self.fc3 = nn.Linear (6,7)
        self.fc4 = nn.Linear (7,4)

    def forward(self, x):
        x = F.relu (self.fc1(x))
        x = F.relu (self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x

model = simple_sd_model()
input_data = torch.rand (10, 1, requires_grad= True)
output_data = model(input_data)
print (output_data)




