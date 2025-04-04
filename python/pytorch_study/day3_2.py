import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
data = pd.read_csv('data.csv')  # Replace 'data.csv' with your dataset
print("Columns in the dataset:", data.columns)  # Check column names

# 2. Preprocess data
X = data['text']  # Assuming your text column is named 'text'
y = data['label']  # Assuming your label column is named 'label'

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert text to numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 3. Define the neural network model
class SentimentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model
input_size = X_train_vectorized.shape[1]
hidden_size = 512
output_size = len(label_encoder.classes_)
model = SentimentModel(input_size, hidden_size, output_size)

# 4. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X_train_vectorized.toarray()))
    loss = criterion(outputs, torch.LongTensor(y_train))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(torch.FloatTensor(X_test_vectorized.toarray()))
    _, predicted = torch.max(test_outputs, 1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted.numpy())
    print(f'Accuracy: {accuracy:.4f}')
    
    # Detailed classification report
    print(classification_report(y_test, predicted.numpy(), target_names=label_encoder.classes_))

# 7. Test the model with new sample inputs
def predict_sentiment(text):
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    with torch.no_grad():
        output = model(torch.FloatTensor(text_vectorized.toarray()))
        _, predicted = torch.max(output, 1)
        return label_encoder.inverse_transform(predicted.numpy())[0]

# Test the model with new sentences
new_samples = [
    "It is very good",
    "Bad",
    "Good",
    "loving you",
    "Loving you",
    "love you",
    "Love you",
    "Very bad",
    "I love you",
    "Fuck",
    "fuck",
    "bad store",
    "i dont love this",
    "not like this"
]

for sample in new_samples:
    sentiment = predict_sentiment(sample)
    print(f'Text: "{sample}" -> Predicted Sentiment: {sentiment}')
