import pandas as pd
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer

#  ۱. داده‌ها را ایجاد و ذخیره کنید
data = {
    "text": [
        "This movie was great",
        "I did not like this movie",
        "The acting was terrible",
        "I loved the plot",
        "It was a boring experience",
        "What a fantastic film!",
        "I hated it",
        "It was okay",
        "Absolutely wonderful!",
        "Not my favorite"
        "Was very good"
        "Very good"
    ],
    "label": [
        1,  # Positive
        0,  # Negative
        0,  # Negative
        1,  # Positive
        0,  # Negative
        1,  # Positive
        0,  # Negative
        0,  # Negative
        1,  # Positive
        0,
        1,
        1   # Negative
    ]
}

# تبدیل دیکشنری به DataFrame و ذخیره در CSV
df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)

#  ۲. خواندن و پردازش داده‌ها
df = pd.read_csv("data.csv")

#  ۳. تبدیل کلمات به اعداد (Tokenization)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"].values

# تبدیل داده‌ها به Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

#  ۴. ساخت مدل شبکه عصبی
class SentimentAnalysisModel(nn.Module):
    def __init__(self, input_size):
        super(SentimentAnalysisModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)  # لایه‌ی مخفی با ۸ نورون
        self.fc2 = nn.Linear(8, 1)  # خروجی (یک مقدار بین ۰ و ۱)
        self.relu = nn.ReLU()  # تابع فعال‌ساز

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # تابع سیگموید برای خروجی بین ۰ و ۱
        return x

#  ۵. تنظیم تابع هزینه و بهینه‌ساز
input_size = X.shape[1]  # تعداد ویژگی‌ها (کلمات منحصر به فرد)
model = SentimentAnalysisModel(input_size)

criterion = nn.BCELoss()  # تابع هزینه برای دسته‌بندی دودویی
optimizer = optim.Adam(model.parameters(), lr=0.01)  # نرخ یادگیری ۰.۰۱

#  ۶. آموزش مدل
epochs = 100  

for epoch in range(epochs):
    # ۱. پیش‌بینی مدل
    y_pred = model(X_tensor)
    
    # ۲. محاسبه‌ی هزینه (Loss)
    loss = criterion(y_pred, y_tensor)
    
    # ۳. پاک کردن گرادیان‌های قبلی
    optimizer.zero_grad()
    
    # ۴. محاسبه‌ی گرادیان‌ها و بروزرسانی وزن‌ها
    loss.backward()
    optimizer.step()

    # ۵. نمایش میزان خطا هر ۱۰ مرحله
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

#  ۷. تست مدل
def predict_sentiment(text):
    # تبدیل متن ورودی به بردار ویژگی‌ها
    text_vectorized = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vectorized, dtype=torch.float32)
    
    # پیش‌بینی مدل
    output = model(text_tensor)
    
    # تبدیل مقدار خروجی به برچسب ۰ یا ۱
    prediction = 1 if output.item() > 0.5 else 0
    
    return "Positive" if prediction == 1 else "Negative"

# 🔹 تست روی چند جمله جدید
print(predict_sentiment("I really enjoyed this movie!"))
print(predict_sentiment("This was the worst experience ever."))
print(predict_sentiment("It was just okay, nothing special."))
print(predict_sentiment("Absolutely loved the storyline!"))
