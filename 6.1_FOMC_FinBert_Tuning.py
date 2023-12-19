import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler,TensorDataset
from transformers import BertTokenizer,BertModel, BertPreTrainedModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CustomFinBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Load the pre-trained BERT model
        self.bert = BertModel(config)

        # Define custom feedforward layers
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)  # Output layer for 3 classes

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get the output from the BERT model
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[1]  # Use pooled output

        # Pass through the custom layers
        x = self.dropout(sequence_output)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc3(x))
        x = self.fc4(x)

        return x

# Prepare the dataset
def encode_texts(tokenizer, texts, max_length=512):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,  # Include token type ids
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        token_type_ids.append(encoded['token_type_ids'])  # Append token type ids

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)  # Concatenate token type ids

    return input_ids, attention_masks, token_type_ids

def train(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        # Move batch to GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch

        model.zero_grad()

        # Forward pass, include token_type_ids
        outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
        loss = nn.CrossEntropyLoss()(outputs, b_labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss

def evaluate(model, validation_dataloader, device):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        # Move batch to GPU
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch

        with torch.no_grad():
            # Forward pass, include token_type_ids
            outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)

        loss = nn.CrossEntropyLoss()(outputs, b_labels)
        total_eval_loss += loss.item()

        preds = torch.argmax(outputs, dim=1).flatten()
        total_eval_accuracy += (preds == b_labels).cpu().numpy().mean()
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    return avg_val_loss, avg_val_accuracy
if __name__ == '__main__':
    # Load the model
    print('Loading The Data')
    model_name = "yiyanghkust/finbert-tone"
    model = CustomFinBERT.from_pretrained(model_name)
    model.to(device)
    # Load the tokenizer
    print("Loading the Tokenizer")
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
    #Load the dataset
    df = pd.read_csv('ft.csv')

    texts = df['sentence'].values
    labels = df['sentiment'].values
    print("Data Encoding")
    # Encode the dataset
    input_ids, attention_masks, token_type_ids = encode_texts(tokenizer, texts)
    labels = torch.tensor(labels)

    # Split the dataset
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.1)
    train_token_types, validation_token_types, _, _ = train_test_split(token_type_ids, labels, random_state=42, test_size=0.1)
    
    # Create the DataLoader
    batch_size = 64
    print("Data Loading")
    train_data = TensorDataset(train_inputs, train_masks, train_token_types, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_token_types, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    # Fine-tune the model
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Number of training epochs
    epochs = 100

    # multi-step learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,80], gamma=0.1, verbose=True)
    check_point_folder = 'models/'
    # Training and evaluation loop
    for epoch in range(epochs):
        print("======== Epoch {:} / {:} ========".format(epoch + 1, epochs))
        avg_train_loss = train(model, train_dataloader, optimizer, device)
        print(f"Average training loss: {avg_train_loss}")

        avg_val_loss, avg_val_acc = evaluate(model, validation_dataloader, device)
        print(f"Validation loss: {avg_val_loss}, Validation Accuracy:{avg_val_acc}")
        scheduler.step()
        # Print validation accuracy if you are calculating it
        if epoch%5 ==0:
            model.save_pretrained(f'models/V1_Epoch{epoch} ')
            checkpoint_path = os.path.join(check_point_folder, f"cp_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)