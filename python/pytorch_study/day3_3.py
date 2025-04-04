import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
# خواندن داده‌ها از فایل
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ایجاد دیکشنری برای تبدیل کاراکترها به اندیس و برعکس
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# تبدیل متن به لیست از اندیس‌ها
data = [char_to_idx[ch] for ch in text]

# تنظیم پارامترهای آموزشی
seq_length = 50  # طول دنباله ورودی
batch_size = 64
hidden_size = 128
num_layers = 2
num_epochs = 100
learning_rate = 0.01
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long),
            torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        )

dataset = TextDataset(data, seq_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

vocab_size = len(chars)
model = LSTMModel(vocab_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    hidden = None  # مقدار اولیه hidden

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs, hidden = model(inputs, hidden)

        # Detach hidden state to avoid graph dependency issues
        hidden = (hidden[0].detach(), hidden[1].detach())

        # Compute loss
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
def generate_text(model, start_text, length=200):
    model.eval()
    input_seq = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    generated_text = start_text

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        next_char_idx = torch.argmax(output[:, -1, :]).item()
        generated_text += idx_to_char[next_char_idx]
        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)

    return generated_text

# تست تولید متن
start_text = "Once upon a time"
print(generate_text(model, start_text, 200))
