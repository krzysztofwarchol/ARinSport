import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import json
import torch
from torch.utils.data import TensorDataset
import time


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 't', 'yes', '1'}:
        return True
    elif value.lower() in {'false', 'f', 'no', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_confusion_matrix(cm,model,head,frames,finetuned,dataset, labels, path):
    en_pl =  {
        "basketball" : "koszykówka",
        "diving": "skoki do wody",
        "volleyball" : "siatkówka",
        "football" : "piłka nożna",
        "aerobic_gymnastics" : "aerobik sportowy",
        "svm" : "svm",
        "randomforest" : "las_losowy",
        "mlp" : "mlp"
    }

    plt.figure(figsize=(18, 15))
    ax = sns.heatmap(cm, square=True, annot=True, fmt='.2f', cbar=True, xticklabels=labels, yticklabels=labels)
    
    plt.ylabel("Prawdziwe etykiety", labelpad=20)
    plt.xlabel("Przewidywane etykiety", labelpad=20)
    
    if finetuned:
        plt.title(f"Macierz pomyłek: dostrojony {model}-{frames}-{en_pl[head]} [{en_pl[dataset].capitalize()}]", pad=20)
    else:
        plt.title(f"Macierz pomyłek: {model}-{frames}-{en_pl[head]} [{en_pl[dataset].capitalize()}]", pad=20)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
    
    plt.tight_layout()
    if finetuned:
        plt.savefig(f"{path}/cm_{dataset}_finetuned_{model}_{head}_{frames}.jpg", dpi=300)
    else:
        plt.savefig(f"{path}/cm_{dataset}_{model}_{head}_{frames}.jpg", dpi=300)
    plt.close()

def average_metrics_for_reports(reports, classes):
    average_metrics = {cls: {'precision': 0, 'recall': 0, 'f1-score': 0} for cls in classes}

    for report in reports:
        for cls in classes:
            average_metrics[cls]['precision'] += report[cls]['precision']
            average_metrics[cls]['recall'] += report[cls]['recall']
            average_metrics[cls]['f1-score'] += report[cls]['f1-score']
    
    num_reports = len(reports)
    for cls in classes:
        average_metrics[cls]['precision'] /= num_reports
        average_metrics[cls]['recall'] /= num_reports
        average_metrics[cls]['f1-score'] /= num_reports

def append_metrics_to_csv(file_path, model, head, frames, finetuned, dataset, acc, acc_std, acc_5, acc_5_std, f1, f1_std):

    row = [model, frames, head, finetuned, acc, acc_std, acc_5, acc_5_std, f1, f1_std, dataset]
    
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['model', 'frames', 'cls_head', 'finetuned', 'acc[mean]', 'acc[std]', 'acc@5[mean]', 'acc@5[std]','f1[mean]', 'f1[std]', 'dataset'])
        
        writer.writerow(row)

def save_metadata(model,dataset,frame,head,finetuned,result):
    
    if finetuned:
        final_path = f"results/model_metric/{model}/{dataset}/{model}_finetuned_{frame}_{head}_{dataset}_metadata.json"
    else:
        final_path = f"results/model_metric/{model}/{dataset}/{model}_{frame}_{head}_{dataset}_metadata.json"
        
    with open(final_path, 'w') as f:
        json.dump(result, f)

def create_dataset(X_train, X_val, X_test, y_train, y_val, y_test):

    X_train_tensor = torch.from_numpy(X_train)
    X_val_tensor = torch.from_numpy(X_val)
    X_test_tensor = torch.from_numpy(X_test)

    y_train_tensor = torch.from_numpy(y_train)
    y_val_tensor = torch.from_numpy(y_val)
    y_test_tensor = torch.from_numpy(y_test)

    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

    return (X_train_tensor, X_val_tensor, X_test_tensor), (y_train_tensor, y_val_tensor, y_test_tensor), (dataset_train, dataset_val)

def start_experiment():
    start_time = time.time()
    return start_time

def end_experiment(start_time,model,frames,head,finetuned,dataset,file_path):
    end_time = time.time()
    total_time_sec = end_time - start_time
    end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    
    hours = int(total_time_sec / 3600)
    total_time_sec %= 3600
    minutes = int(total_time_sec / 60)
    seconds = total_time_sec % 60
    
    print(f"Time: {hours}:{minutes}:{seconds:.2f}")

    row = [model, frames, head, finetuned, dataset, f"{hours}:{minutes}:{seconds}", end_time_formatted]
    
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['model', 'frames', 'cls_head', 'finetuned', 'dataset','time_exp','timestamp'])
        
        writer.writerow(row)



    