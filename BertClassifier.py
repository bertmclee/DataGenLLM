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









# def bert_classifer_training(train_data, evaluate_data, label_col, model_path):



#     # Save Model
    
#     return trained_model


# def bert_classifier_evaluate(model, dataset, label_col):

#     return






# train_df = pd.read_csv('/content/drive/My Drive/FM_Final_Proj_Code_Repo/Oracle_Datasets/sst2_train.csv')
# validation_df = pd.read_csv('/content/drive/My Drive/FM_Final_Proj_Code_Repo/Oracle_Datasets/sst2_validation.csv')

# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def tokenize_and_encode(sentences):
#     input_ids = []
#     attention_masks = []

#     for sent in sentences:
#         encoded_dict = tokenizer.encode_plus(
#             sent,                      # Sentence to encode
#             add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
#             max_length=64,            # Pad & truncate all sentences
#             pad_to_max_length=True,
#             return_attention_mask=True, # Construct attention masks
#             return_tensors='pt',       # Return pytorch tensors
#         )

#         input_ids.append(encoded_dict['input_ids'])
#         attention_masks.append(encoded_dict['attention_mask'])

#     return input_ids, attention_masks

# train_inputs, train_masks = tokenize_and_encode(train_df['sentence'].values)
# validation_inputs, validation_masks = tokenize_and_encode(validation_df['sentence'].values)
# test_inputs, test_masks = tokenize_and_encode(test_df['sentence'].values)


# import torch

# train_labels = torch.tensor(train_df['label'].values)
# validation_labels = torch.tensor(validation_df['label'].values)
# test_labels = torch.tensor(test_df['label'].values)


# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# batch_size = 32  # You can modify the batch size as needed

# # Create the DataLoader for training set
# train_data = TensorDataset(torch.cat(train_inputs, dim=0), torch.cat(train_masks, dim=0), train_labels)
# train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# # Do similar for validation and test sets
# validation_data = TensorDataset(torch.cat(validation_inputs, dim=0), torch.cat(validation_masks, dim=0), validation_labels)
# validation_sampler = RandomSampler(validation_data)
# validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# test_data = TensorDataset(torch.cat(test_inputs, dim=0), torch.cat(test_masks, dim=0), test_labels)
# test_sampler = RandomSampler(test_data)
# test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


# from transformers import BertForSequenceClassification, AdamW

# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
#     num_labels=2,        # Binary classification (0 or 1)
#     output_attentions=False,  # Whether the model returns attentions weights.
#     output_hidden_states=False, # Whether the model returns all hidden-states.
# )


# from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup

# # Parameters:
# lr = 2e-5
# adam_epsilon = 1e-8

# # Number of training epochs (authors recommend between 2 and 4)
# epochs = 3

# num_warmup_steps = 0
# num_training_steps = len(train_dataloader)*epochs

# ### In Transformers, optimizer and schedules are splitted and instantiated like this:
# optimizer = AdamW(model.parameters(), lr=lr,eps=adam_epsilon,correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

# # Check if CUDA is available (for GPU usage)
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())
#     print('We will use the GPU:', torch.cuda.get_device_name(0))

# else:
#     device = torch.device("cpu")
#     print('No GPU available, using the CPU instead.')

# # Move the model to the chosen device
# model.to(device)


# from tqdm import trange
# import numpy as np # linear algebra
# from sklearn.metrics import accuracy_score,matthews_corrcoef
# from tqdm import tqdm, trange,tnrange,tqdm_notebook

# ## Store our loss and accuracy for plotting
# train_loss_set = []
# learning_rate = []

# # Gradients gets accumulated by default
# model.zero_grad()

# # tnrange is a tqdm wrapper around the normal python range
# for _ in tnrange(1,epochs+1,desc='Epoch'):
#   print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
#   # Calculate total loss for this epoch
#   batch_loss = 0

#   for step, batch in enumerate(train_dataloader):
#     # Set our model to training mode (as opposed to evaluation mode)
#     model.train()

#     # Add batch to GPU
#     batch = tuple(t.to(device) for t in batch)
#     # Unpack the inputs from our dataloader
#     b_input_ids, b_input_mask, b_labels = batch

#     # Forward pass
#     outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
#     loss = outputs[0]

#     # Backward pass
#     loss.backward()

#     # Clip the norm of the gradients to 1.0
#     # Gradient clipping is not in AdamW anymore
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#     # Update parameters and take a step using the computed gradient
#     optimizer.step()

#     # Update learning rate schedule
#     scheduler.step()

#     # Clear the previous accumulated gradients
#     optimizer.zero_grad()

#     # Update tracking variables
#     batch_loss += loss.item()

#   # Calculate the average loss over the training data.
#   avg_train_loss = batch_loss / len(train_dataloader)

#   #store the current learning rate
#   for param_group in optimizer.param_groups:
#     print("\n\tCurrent Learning rate: ",param_group['lr'])
#     learning_rate.append(param_group['lr'])

#   train_loss_set.append(avg_train_loss)
#   print(F'\n\tAverage Training loss: {avg_train_loss}')

#   # Validation

#   # Put model in evaluation mode to evaluate loss on the validation set
#   model.eval()

#   # Tracking variables
#   eval_accuracy,eval_mcc_accuracy,nb_eval_steps = 0, 0, 0

#   # Evaluate data for one epoch
#   for batch in validation_dataloader:
#     # Add batch to GPU
#     batch = tuple(t.to(device) for t in batch)
#     # Unpack the inputs from our dataloader
#     b_input_ids, b_input_mask, b_labels = batch
#     # Telling the model not to compute or store gradients, saving memory and speeding up validation
#     with torch.no_grad():
#       # Forward pass, calculate logit predictions
#       logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

#     # Move logits and labels to CPU
#     logits = logits[0].to('cpu').numpy()
#     label_ids = b_labels.to('cpu').numpy()

#     pred_flat = np.argmax(logits, axis=1).flatten()
#     labels_flat = label_ids.flatten()

#     df_metrics=pd.DataFrame({'Epoch':epochs,'Actual_class':labels_flat,'Predicted_class':pred_flat})

#     tmp_eval_accuracy = accuracy_score(labels_flat,pred_flat)
#     tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)

#     eval_accuracy += tmp_eval_accuracy
#     eval_mcc_accuracy += tmp_eval_mcc_accuracy
#     nb_eval_steps += 1

#   print(F'\n\tValidation Accuracy: {eval_accuracy/nb_eval_steps}')
#   print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy/nb_eval_steps}')

  
# model_save_path = "/content/drive/My Drive/FM_Final_Proj_Code_Repo/Trained_Models"
# os.makedirs(model_save_path, exist_ok=True)

# # After or during training, depending on when you want to save
# model_save_name = "sst2_bert_model_2.pt"
# save_path = os.path.join(model_save_path, model_save_name)
# torch.save(model.state_dict(), save_path)
# print(f"Model saved to {save_path}")
