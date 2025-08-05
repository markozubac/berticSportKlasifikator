# Load model directly
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter

# Load BERTić model and tokenizer
model_name = "classla/bcms-bertic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name, force_download=True, output_hidden_states=True)

print(bert_model)

for param in bert_model.parameters():
    param.requires_grad = False

class CustomBertClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim=128):
        super(CustomBertClassifier, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(768, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # Extract pooled output (CLS token representation)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # CLS token embedding

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load dataset
data = pd.read_csv("classla_LAT_anotacija4_24.10_original.csv")

class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.float)
        }

# Create dataset and dataloader
dataset = CustomDataset(data["sentence"], data["output"], tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = CustomBertClassifier(bert_model)

optimizer = AdamW(model.parameters(), lr=1e-4)
loss_fn = BCEWithLogitsLoss()

epochs = 100
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model.to(device)
print(device)

writer = SummaryWriter("runs/bertic_experiment")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask).squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar("Training Loss", loss.item(), epoch * len(dataloader) + batch_idx)
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
writer.close()

def predict_sentence(model, tokenizer, sentence, device):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            sentence,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        output = model(input_ids, attention_mask).squeeze()
        prediction = torch.sigmoid(output).item() > 0.5
        return int(prediction)

example_sentence = "Teniski meč će biti na programu između dva najbolja igrača."
print(f"Sentence: {example_sentence}")
predicted_label = predict_sentence(model, tokenizer, example_sentence, device)
print(f"Predicted Label: {predicted_label}")
example_sentence = "Dragisa je nrvozan zbog gluposti na poslu."
print(f"Sentence: {example_sentence}")
predicted_label = predict_sentence(model, tokenizer, example_sentence, device)
print(f"Predicted Label: {predicted_label}")
example_sentence = "Točak je doktor za rokenrol."
print(f"Sentence: {example_sentence}")
predicted_label = predict_sentence(model, tokenizer, example_sentence, device)
print(f"Predicted Label: {predicted_label}")
example_sentence = "Vincan je izgubio stonog tenisa, ali će KK Crvena Zvezda noćas pobediti."
print(f"Sentence: {example_sentence}")
predicted_label = predict_sentence(model, tokenizer, example_sentence, device)
print(f"Predicted Label: {predicted_label}")
example_sentence = "KK Crvena Zvezda je na domaćem terenu pobedila ekipu Asvela rezultatom 73:66."
print(f"Sentence: {example_sentence}")
predicted_label = predict_sentence(model, tokenizer, example_sentence, device)
print(f"Predicted Label: {predicted_label}")

# Evaluate the model on the test dataset
def evaluate_model(model, tokenizer, test_file, device):
    model.eval()
    test_data = pd.read_csv(test_file)
    test_dataset = CustomDataset(test_data["sentence"], test_data["output"], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask).squeeze()
            predictions = torch.sigmoid(outputs) > 0.5

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Run evaluation
test_file = "test.csv"
evaluate_model(model, tokenizer, test_file, device)
