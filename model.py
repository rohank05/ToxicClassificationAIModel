import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import pickle

class ToxicityDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[["comment_text", "toxicity", "threat", "severe_toxicity", "obscene", "insult", "identity_attack", "sexual_explicit"]]
        self.data = self.data.dropna()
        self.texts = self.data["comment_text"].values
        self.labels = self.data.drop(["comment_text"], axis=1).values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            add_special_tokens=True, 
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.FloatTensor(label)

# Load the data and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_data = ToxicityDataset("all_data.csv", tokenizer, max_length=400)

# Set up the DataLoader
batch_size = 30
num_workers = 2
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.nn.Sequential(
    torch.nn.Embedding(tokenizer.vocab_size, 20),
    torch.nn.Flatten(),
    torch.nn.Linear(20*400, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 7),
    torch.nn.Sigmoid()
)
model.to(device)
print(f"Using {device}")
# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, batch_data in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch_data
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print('Epoch {}/{} | Batch {}/{} | Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(train_dataloader), loss.item()), end='\r')
    print()  

# Save the tokenizer and model
torch.save(model.state_dict(), "model.pt")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

