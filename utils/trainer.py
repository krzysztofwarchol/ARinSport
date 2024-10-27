import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

def train(model, 
          criterion, 
          optimizer, 
          EPOCHS, 
          train_dataloader, 
          val_dataloader, 
          device,
          print_metric = False):
    
    histry = {'epochs' : [],
              'loss' : [],
              'acc_train' : [],
              'acc_val' : [],
              'f1_train': [],
              'f1_val' : []}

    best_valid_f1 = 0.0
    best_valid_acc = 0.0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        model.train()
        
        for inputs, labels in train_dataloader:
    
            labels = labels.type(torch.LongTensor)
            
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = model(inputs.float())
    
            loss = criterion(outputs, labels)
    
            loss.backward()
    
            optimizer.step()
    
            total_loss += loss.item()
    
            _, predicted = torch.max(outputs, 1)
    
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        accuracy_train = accuracy_score(all_labels, all_predictions)
        f1_train = f1_score(all_labels, all_predictions, average='macro')

        all_labels = []
        all_predictions = []

        model.eval()

        for inputs, labels in val_dataloader:
            
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs.float())
            
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        accuracy_val = accuracy_score(all_labels, all_predictions)
        f1_val = f1_score(all_labels, all_predictions, average='macro')


        # if accuracy_val > best_valid_acc:
        #     best_valid_acc = accuracy_val
        #     torch.save(model.state_dict(), 'saved_weights.pt')
        
        histry['epochs'].append(epoch)
        histry['acc_train'].append(accuracy_train)
        histry['acc_val'].append(accuracy_val)
        histry['f1_train'].append(f1_train)
        histry['f1_val'].append(f1_val)
        histry['loss'].append(total_loss)

        if print_metric:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Acc_train: {accuracy_train :.4f}, F1_train: {f1_train:.4f} || Acc_val: {accuracy_val :.4f}, F1_val: {f1_val:.4f}')
    
    return histry

def finetune(model, 
          criterion, 
          optimizer, 
          EPOCHS, 
          train_dataloader, 
          val_dataloader, 
          device,
          print_metric = False):
    
    histry = {'epochs' : [],
              'loss' : [],
              'acc_train' : [],
              'acc_val' : [],
              'f1_train': [],
              'f1_val' : []}

    best_valid_f1 = 0.0
    best_valid_acc = 0.0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        model.train()
        
        for inputs, labels in tqdm(train_dataloader,total=len(train_dataloader.dataset)):
    
            labels = labels.type(torch.LongTensor)
            
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = model(inputs.float())
    
            loss = criterion(outputs, labels)
    
            loss.backward()
    
            optimizer.step()
    
            total_loss += loss.item()
    
            _, predicted = torch.max(outputs, 1)
    
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        accuracy_train = accuracy_score(all_labels, all_predictions)
        f1_train = f1_score(all_labels, all_predictions, average='macro')

        all_labels = []
        all_predictions = []

        model.eval()

        for inputs, labels in tqdm(val_dataloader,total=len(val_dataloader.dataset)):
            
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs.float())
            
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        accuracy_val = accuracy_score(all_labels, all_predictions)
        f1_val = f1_score(all_labels, all_predictions, average='macro')


        if accuracy_val > best_valid_acc:
            best_valid_acc = accuracy_val
            torch.save(model.state_dict(), f"results/finetuned_models/best_model/best_model_{model.model_type}_{model.dataset}_epoch_{epoch}_acc_{round(accuracy_val,3)}_f1_{round(f1_val,3)}.pt")

        if (epoch + 1) == EPOCHS:
            torch.save(model.state_dict(), f"results/finetuned_models/finetuned_model_{model.model_type}_{model.dataset}.pt")   
        
        histry['epochs'].append(epoch)
        histry['acc_train'].append(accuracy_train)
        histry['acc_val'].append(accuracy_val)
        histry['f1_train'].append(f1_train)
        histry['f1_val'].append(f1_val)
        histry['loss'].append(total_loss)

        if print_metric:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Acc_train: {accuracy_train :.4f}, F1_train: {f1_train:.4f} || Acc_val: {accuracy_val :.4f}, F1_val: {f1_val:.4f}')
    
    return histry


def evaluation(model,test_data,test_target,y_train,print_result: bool = False):
    
    test_data = test_data.float()
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
        outputs = torch.argmax(predictions, dim=1)

    cm = confusion_matrix(test_target.long(),outputs)
    accuracy = accuracy_score(test_target.long(), outputs)
    f1 = f1_score(test_target.long(), outputs, average='macro')
    acc_top_5 = top_k_accuracy_score(test_target.long(), predictions,k=5,labels=np.unique(y_train.float()))
    report = classification_report(test_target.long(),outputs,output_dict=True)

    if print_result:
        print(f"Acc:{accuracy}")
        print(f"Acc@5:{acc_top_5}")
        print(f"F1_score:{f1}")

    return accuracy, acc_top_5, f1, cm, report