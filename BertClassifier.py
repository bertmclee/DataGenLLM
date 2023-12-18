import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
# from torch.optim import AdamW  # Use PyTorch's AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    

def tokenize_and_encode(sentences, tokenizer, max_length=64):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,                      
            add_special_tokens=True,   
            max_length=max_length,
            padding='max_length', 
            return_attention_mask=True, 
            return_tensors='pt',
            truncation=True,  # Add this line
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks


def create_dataloader(inputs, masks, labels, batch_size=32, seed=42):
    # Setting the seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    data = TensorDataset(torch.cat(inputs, dim=0), torch.cat(masks, dim=0), labels)
    sampler = RandomSampler(data, generator=generator)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def bert_classifier_training(model, train_dataloader, validation_dataloader, optimizer, scheduler, device, epochs=3):
    best_accuracy = 0
    best_model_state = None  # To store the state of the best model

    for epoch in range(epochs):
        print("<" + "="*22 + F" Epoch {epoch+1} "+ "="*22 + ">")
        
        # Training Phase
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(F'\n\tAverage Training loss: {avg_train_loss}')

        # Evaluation Phase
        model.eval()
        eval_accuracy, nb_eval_steps = 0, 0
        total_precision, total_recall, total_f1 = 0, 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0].to('cpu').numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()

            eval_accuracy += accuracy_score(labels_flat, pred_flat)
            precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, pred_flat, \
                                                                        average='weighted', zero_division=1)
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            nb_eval_steps += 1

        avg_accuracy = eval_accuracy / nb_eval_steps
        avg_precision = total_precision / nb_eval_steps
        avg_recall = total_recall / nb_eval_steps
        avg_f1 = total_f1 / nb_eval_steps

        print(F'\n\tValidation Accuracy: {avg_accuracy}')
        print(F'\n\tValidation Precision: {avg_precision}')
        print(F'\n\tValidation Recall: {avg_recall}')
        print(F'\n\tValidation F1-Score: {avg_f1}')

        # Check if this is the best model so far
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_model_state = model.state_dict().copy()  # Copy the model state
            print(f'New best model found at epoch {epoch+1} with accuracy: {avg_accuracy}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)  # Load the best model state
        print('Loaded the best model')

    return model


def bert_classifier_evaluate(model, validation_dataloader, device):
    model.eval()
    eval_accuracy, nb_eval_steps = 0, 0
    total_precision, total_recall, total_f1 = 0, 0, 0
    all_preds, all_labels = [], []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = logits[0].to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()

        preds = np.argmax(logits, axis=1).flatten()
        all_preds.extend(preds)
        all_labels.extend(label_ids)

        eval_accuracy += accuracy_score(label_ids, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(label_ids, preds, average='weighted', zero_division=1)
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        nb_eval_steps += 1

    avg_accuracy = eval_accuracy / nb_eval_steps
    avg_precision = total_precision / nb_eval_steps
    avg_recall = total_recall / nb_eval_steps
    avg_f1 = total_f1 / nb_eval_steps

    # Plotting the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')  # Using seaborn for better visualization
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return avg_accuracy, avg_precision, avg_recall, avg_f1



def save_model(model, save_path, model_name):
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



def bert_main(train_df, eval_df, model_path, model_name, text_col='sentence', label_col='label', batch_size=32, epochs=3, max_length=64, lr=2e-5, adam_epsilon=1e-8):
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and encode datasets
    train_inputs, train_masks = tokenize_and_encode(train_df[text_col].values, tokenizer, max_length)
    eval_inputs, eval_masks = tokenize_and_encode(eval_df[text_col].values, tokenizer, max_length)

    # Convert labels to tensors
    train_labels = torch.tensor(train_df[label_col].values)
    eval_labels = torch.tensor(eval_df[label_col].values)

    # Create dataloaders
    train_dataloader = create_dataloader(train_inputs, train_masks, train_labels, batch_size)
    eval_dataloader = create_dataloader(eval_inputs, eval_masks, eval_labels, batch_size)

    # Initialize BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(train_df[label_col].unique()))

    # Setting up the device for GPU usage
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('Using CPU.')

    model.to(device)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, eps=adam_epsilon)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train and evaluate
    model = bert_classifier_training(model, train_dataloader, eval_dataloader, optimizer, scheduler, device, epochs)

    # Evaluation on evaluation dataset
    eval_accuracy, eval_precision, eval_recall, eval_f1 = bert_classifier_evaluate(model, eval_dataloader, device)
    print(f'\nEvaluation Accuracy: {eval_accuracy}')
    print(f'Evaluation Precision: {eval_precision}')
    print(f'Evaluation Recall: {eval_recall}')
    print(f'Evaluation F1-Score: {eval_f1}')

    # Save the model
    save_model(model, model_path, model_name)




