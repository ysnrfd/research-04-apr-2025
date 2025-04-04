import torch
import torch.nn as nn
import torch.optim as optim

# ⚡ 1. تعریف شبکه عصبی
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)  # لایه پنهان
        self.activation = nn.ReLU()  # تابع فعال‌سازی
        self.output = nn.Linear(hidden_size, output_size)  # لایه خروجی

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

# ⚡ 2. تنظیمات مدل
input_size = 10   # تعداد ویژگی‌های ورودی (مثلاً 4 ویژگی برای هر داده)
hidden_size = 4 # تعداد نورون‌های لایه پنهان
output_size = 4  # تعداد کلاس‌ها (مثلاً 3 کلاس)

model = SimpleNN(input_size, hidden_size, output_size)
print(model)

# ⚡ 3. تعریف تابع هزینه و بهینه‌ساز
criterion = nn.CrossEntropyLoss()  # مناسب برای مسائل طبقه‌بندی
optimizer = optim.Adam(model.parameters(), lr=0.01)  # الگوریتم بهینه‌سازی

# ⚡ 4. داده‌های ورودی ساختگی
X_train = torch.rand(10, input_size)  # 5 نمونه، هر نمونه دارای 4 ویژگی
y_train = torch.tensor([0, 1, 2, 1, 0, 1, 2, 1, 0, 1])  # برچسب‌های کلاس (0، 1 یا 2)

# ⚡ 5. آموزش مدل (یک epoch برای مثال)
for i in range(1):
    optimizer.zero_grad()  # تنظیم گرادیان‌ها به صفر
    outputs = model(X_train)  # عبور داده‌ها از شبکه
    loss = criterion(outputs, y_train)  # محاسبه خطا
    loss.backward()  # محاسبه گرادیان‌ها
    optimizer.step()  # بروزرسانی وزن‌ها

# ⚡ 6. نمایش خروجی و خطا
print(f"Output:\n{outputs}")
print(f"Loss: {loss.item()}")
