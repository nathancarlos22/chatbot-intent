import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import deque

# Verificar se o dispositivo está disponível
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Configurações
class Config:
    num_classes = 10  # Ajuste com base no número de intenções
    epochs = 15
    batch_size = 8
    lr = 1e-5
    max_length = 15

cfg = Config()

# Carregar o arquivo JSON
with open("intent.json", "r") as f:
    data = json.load(f)

# Extrair intenções e textos
df_intents = data['intents'][:cfg.num_classes]  # Usar apenas as primeiras 10 intenções
texts = []
labels = []
responses = {}

# Mapear intenções para labels
intent2label = {}
label2intent = {}
for idx, intent_data in enumerate(df_intents):
    intent = intent_data['intent']
    intent2label[intent] = idx
    label2intent[idx] = intent
    for text in intent_data['text']:
        texts.append(text)
        labels.append(idx)
    # Coletar respostas para cada intenção
    responses[intent] = intent_data['responses']

# Criar um DataFrame (opcional, para visualização)
import pandas as pd
df = pd.DataFrame({'text': texts, 'label': labels})

# Definir a classe Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(texts, padding='max_length', max_length=cfg.max_length, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Dividir os dados em conjuntos de treinamento e validação
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Criar objetos Dataset
train_dataset = IntentDataset(train_texts, train_labels)
val_dataset = IntentDataset(val_texts, val_labels)

# Definir o modelo
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, cfg.num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Usar o pooler_output
        dropout_output = self.dropout(pooled_output)
        output = self.linear(dropout_output)
        return output

# Caminho para o modelo salvo
model_path = 'intent_model.pth'

# Função para treinamento
def train(model, train_dataset, val_dataset, learning_rate, epochs):
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        model.train()
        total_acc_train = 0
        total_loss_train = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss_train += loss.item()
            acc = (outputs.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()

        model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item()
                acc = (outputs.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"Train Loss: {total_loss_train / len(train_dataset):.3f}, Train Accuracy: {total_acc_train / len(train_dataset):.3f}")
        print(f"Val Loss: {total_loss_val / len(val_dataset):.3f}, Val Accuracy: {total_acc_val / len(val_dataset):.3f}")

# Verificar se o modelo salvo existe
model = BertClassifier()
if os.path.exists(model_path):
    print("Carregando modelo salvo...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Modelo não encontrado. Iniciando o treinamento...")
    train(model, train_dataset, val_dataset, cfg.lr, cfg.epochs)
    torch.save(model.state_dict(), model_path)
    print("Modelo treinado e salvo.")

model.to(device)
model.eval()

# Função de predição
def predict_intent(model, text):
    with torch.no_grad():
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', max_length=cfg.max_length, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_label = torch.max(probs, dim=1)
        if confidence.item() < 0.3:
            return None
        intent = label2intent[predicted_label.item()]
        return intent

# Função para substituir <HUMAN> pelo nome do usuário
def personalize_response(response, user_name):
    return response.replace("<HUMAN>", user_name)

# Funcionalidade do chatbot
def main():
    global model
    user_name = ""
    intents_history = deque(maxlen=3)

    print("Olá! Eu sou o seu assistente virtual.")
    if not user_name:
        user_name = input("Por favor, qual é o seu nome? ").strip()
        print(f"Prazer em conhecê-lo, {user_name}!")

    while True:
        user_input = input(f"{user_name}: ").strip()
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando a conversa. Até logo!")
            break
        elif user_input.lower() == 'consultar intenções':
            if intents_history:
                print("Últimas três intenções detectadas:")
                for idx, intent in enumerate(list(intents_history)[::-1], 1):
                    print(f"{idx}. {intent}")
            else:
                print("Nenhuma intenção detectada até o momento.")
            continue

        intent = predict_intent(model, user_input)
        if intent:
            intents_history.append(intent)
            # Gerar uma resposta aleatória das disponíveis para a intenção
            response = random.choice(responses[intent])
            # Personalizar a resposta substituindo <HUMAN> pelo nome do usuário
            personalized_response = personalize_response(response, user_name)
            print(f"{user_name}, sua intenção é: {intent}")
            print(f"Resposta: {personalized_response}")
        else:
            print(f"Desculpe, {user_name}, não consegui identificar sua intenção.")

if __name__ == "__main__":
    main()
