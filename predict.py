import torch
import pickle
from transformers import AutoTokenizer

# Load the saved tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the saved model
model = torch.nn.Sequential(
    torch.nn.Embedding(tokenizer.vocab_size, 20),
    torch.nn.Flatten(),
    torch.nn.Linear(20*400, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 7),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load("model.pt"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def predict(text):
    labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
    encoding = tokenizer.encode_plus(
        text,
        max_length=400,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].squeeze()
    attention_mask = encoding['attention_mask'].squeeze()

    # Set the model to evaluation mode
    model.eval()

    # Move the input to the same device as the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        outputs = torch.round(outputs).squeeze().tolist()

    # Map the output to their respective labels
    result = {}
    for i in range(len(labels)):
        result[labels[i]] = outputs[i]

    # Return the predicted labels
    return result

