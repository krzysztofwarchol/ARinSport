import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from utils.models import MLP
from utils.trainer import train, evaluation
from utils.utils import average_metrics_for_reports, save_confusion_matrix, append_metrics_to_csv
import warnings
warnings.filterwarnings('always')


def default_mlp_params():
    result = {
        "default_params" : {
            "batch_size" : 64,
            "lr" : 0.001,
            "dropout" : 0,
            "epochs" : 100
        }
    }
    return result

def search_batch_size(dataset_train, 
                      dataset_val,
                      device,
                      input_dim, 
                      class_size,
                      lr: float = 0.001, 
                      dropout_ratio: float = 0, 
                      epochs: int = 100, 
                      iterration: int = 5,
                      print_result: bool = False):
    
    result = {
        "batch_size" : {
            "itteration" : {
                "param" : [],
                "acc" : [],
                "f1" : [],
                "histry" : []
            },
            "result_mean": {
                'param' : [],
                'acc_mean' : [],
                'acc_std' : [],
                'f1_mean' : [],
                'f1_std' : [],
            },
            "best_result" : {
                "best_param" : "",
                "acc_mean" : 0.0,
                "acc_std" : 0.0,
                "f1_mean" : 0.0,
                "f1_std" : 0.0,
            }
        }
    }

    for _ in range(iterration):
        for batch_size in [8,16,32,64,128]:

            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
            val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

            model = MLP(input_dim=input_dim, dropout_ratio=dropout_ratio, class_size=class_size)
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            histry = train(model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            EPOCHS=epochs,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            device=device)
            
            hist = pd.DataFrame(histry)
            
            result["batch_size"]["itteration"]["param"].append(batch_size)
            result["batch_size"]["itteration"]["acc"].append(hist.tail(1).acc_val.values[0])
            result["batch_size"]["itteration"]["f1"].append(hist.tail(1).f1_val.values[0])
            result["batch_size"]["itteration"]["histry"].append(histry)


    tmp = pd.DataFrame(result["batch_size"]["itteration"])

    for param in tmp.param.unique():
        result["batch_size"]["result_mean"]['param'].append(int(param))
        result["batch_size"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["batch_size"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["batch_size"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["batch_size"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["batch_size"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["batch_size"]["best_result"]['best_param'] = int(best_param)
    result["batch_size"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["batch_size"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["batch_size"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["batch_size"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param batch_size]: {best_param}, Acc: {result['batch_size']['best_result']['acc_mean']} +- {result['batch_size']['best_result']['acc_std']}, F1: {result['batch_size']['best_result']['f1_mean']} +- {result['batch_size']['best_result']['f1_std']}")

    return int(best_param), result


def search_lr(dataset_train, 
                dataset_val,
                device,
                input_dim, 
                class_size,
                batch_size: int = 64,
                dropout_ratio: float = 0, 
                epochs: int = 100, 
                iterration: int = 5,
                print_result: bool = False):

    result = {
        "lr" : {
            "itteration" : {
                "param" : [],
                "acc" : [],
                "f1" : [],
                "histry" : [] 
            },
            "result_mean": {
                'param' : [],
                'acc_mean' : [],
                'acc_std' : [],
                'f1_mean' : [],
                'f1_std' : [],
            },
            "best_result" : {
                "best_param" : "",
                "acc_mean" : 0.0,
                "acc_std" : 0.0,
                "f1_mean" : 0.0,
                "f1_std" : 0.0,
            }
        }
    }

    for _ in range(iterration):
        for lr in [0.01,0.03,0.05,0.07,0.09,
                   0.001,0.003,0.005,0.007,0.009,
                   0.00001,0.00003,0.00005,0.00007,0.00009]:

            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
            val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

            model = MLP(input_dim=input_dim, dropout_ratio=dropout_ratio, class_size=class_size)
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            histry = train(model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            EPOCHS=epochs,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            device=device)
            
            hist = pd.DataFrame(histry)
            
            result["lr"]["itteration"]["param"].append(lr)
            result["lr"]["itteration"]["acc"].append(hist.tail(1).acc_val.values[0])
            result["lr"]["itteration"]["f1"].append(hist.tail(1).f1_val.values[0])
            result["lr"]["itteration"]["histry"].append(histry)


    tmp = pd.DataFrame(result["lr"]["itteration"])

    for param in tmp.param.unique():
        result["lr"]["result_mean"]['param'].append(param)
        result["lr"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["lr"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["lr"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["lr"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["lr"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["lr"]["best_result"]['best_param'] = best_param
    result["lr"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["lr"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["lr"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["lr"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param lr]: {best_param}, Acc: {result['lr']['best_result']['acc_mean']} +- {result['lr']['best_result']['acc_std']}, F1: {result['lr']['best_result']['f1_mean']} +- {result['lr']['best_result']['f1_std']}")

    return best_param, result

def search_dropout(dataset_train, 
                dataset_val,
                device,
                input_dim, 
                class_size,
                batch_size: int = 64,
                lr: float = 0.001, 
                epochs: int = 100, 
                iterration: int = 5,
                print_result: bool = False):

    result = {
        "dropout" : {
            "itteration" : {
                "param" : [],
                "acc" : [],
                "f1" : [],
                "histry" : [] 
            },
            "result_mean": {
                'param' : [],
                'acc_mean' : [],
                'acc_std' : [],
                'f1_mean' : [],
                'f1_std' : [],
            },
            "best_result" : {
                "best_param" : "",
                "acc_mean" : 0.0,
                "acc_std" : 0.0,
                "f1_mean" : 0.0,
                "f1_std" : 0.0,
            }
        }
    }

    for _ in range(iterration):
        for dropout in [0.0,0.2,0.4,0.6,0.8,1.0]:

            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
            val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

            model = MLP(input_dim=input_dim, dropout_ratio=dropout, class_size=class_size)
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            histry = train(model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            EPOCHS=epochs,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            device=device)
            
            hist = pd.DataFrame(histry)
            
            result["dropout"]["itteration"]["param"].append(dropout)
            result["dropout"]["itteration"]["acc"].append(hist.tail(1).acc_val.values[0])
            result["dropout"]["itteration"]["f1"].append(hist.tail(1).f1_val.values[0])
            result["dropout"]["itteration"]["histry"].append(histry)


    tmp = pd.DataFrame(result["dropout"]["itteration"])

    for param in tmp.param.unique():
        result["dropout"]["result_mean"]['param'].append(param)
        result["dropout"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["dropout"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["dropout"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["dropout"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["dropout"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["dropout"]["best_result"]['best_param'] = best_param
    result["dropout"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["dropout"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["dropout"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["dropout"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param dropout]: {best_param}, Acc: {result['dropout']['best_result']['acc_mean']} +- {result['dropout']['best_result']['acc_std']}, F1: {result['dropout']['best_result']['f1_mean']} +- {result['dropout']['best_result']['f1_std']}")

    return best_param, result

def search_epochs(dataset_train, 
                dataset_val,
                device,
                input_dim, 
                class_size,
                batch_size: int = 64,
                lr: float = 0.001, 
                dropout_ratio: float = 0, 
                iterration: int = 5,
                print_result: bool = False):

    result = {
        "epochs" : {
            "itteration" : {
                "param" : [],
                "acc" : [],
                "f1" : [],
                "histry" : [] 
            },
            "result_mean": {
                'param' : [],
                'acc_mean' : [],
                'acc_std' : [],
                'f1_mean' : [],
                'f1_std' : [],
            },
            "best_result" : {
                "best_param" : "",
                "acc_mean" : 0.0,
                "acc_std" : 0.0,
                "f1_mean" : 0.0,
                "f1_std" : 0.0,
            }
        }
    }

    for _ in range(iterration):
        for epochs in [50,100,150,200,250,300]:

            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
            val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

            model = MLP(input_dim=input_dim, dropout_ratio=dropout_ratio, class_size=class_size)
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            histry = train(model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            EPOCHS=epochs,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            device=device)
            
            hist = pd.DataFrame(histry)
            
            result["epochs"]["itteration"]["param"].append(epochs)
            result["epochs"]["itteration"]["acc"].append(hist.tail(1).acc_val.values[0])
            result["epochs"]["itteration"]["f1"].append(hist.tail(1).f1_val.values[0])
            result["epochs"]["itteration"]["histry"].append(histry)


    tmp = pd.DataFrame(result["epochs"]["itteration"])

    for param in tmp.param.unique():
        result["epochs"]["result_mean"]['param'].append(int(param))
        result["epochs"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["epochs"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["epochs"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["epochs"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["epochs"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["epochs"]["best_result"]['best_param'] = int(best_param)
    result["epochs"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["epochs"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["epochs"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["epochs"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param epochs]: {best_param}, Acc: {result['epochs']['best_result']['acc_mean']} +- {result['epochs']['best_result']['acc_std']}, F1: {result['epochs']['best_result']['f1_mean']} +- {result['epochs']['best_result']['f1_std']}")

    return int(best_param), result

def test_best_param_mlp(dataset_train, 
                        dataset_val,
                        X_test_tensor,
                        y_test_tensor,
                        y_train_tensor,
                        y_test,
                        device,
                        input_dim, 
                        class_size,
                        model_name,
                        frames,
                        finetuned,
                        dataset,
                        encoder,
                        batch_size: int = 64,
                        lr: float = 0.001, 
                        dropout_ratio: float = 0,
                        epochs: int = 100,
                        iterration: int = 5,
                        print_result: bool = False):
    
    result = {
        "test_results" : {
            "itteration" : {
                "acc" : [],
                "acc@5" : [],
                "f1" : [],
                "cm" : [],
                "reports": [],
                "histry" : []
            },
            "result": {
                'acc_mean' : 0.0,
                'acc_std' : 0.0,
                'acc@5_mean' : 0.0,
                'acc@5_std' : 0.0,
                'f1_mean' : 0.0,
                'f1_std' : 0.0,
                'cm_mean' : None,
                'reports_mean' : None
            }
        }
    }

    cm_list = []

    for i in range(iterration):
        train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        model = MLP(input_dim=input_dim, dropout_ratio=dropout_ratio, class_size=class_size)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        histry = train(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        EPOCHS=epochs,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        device=device)
        
        acc, acc_5, f1, cm, report = evaluation(model.cpu(),X_test_tensor,y_test_tensor,y_train_tensor)

        result["test_results"]["itteration"]["acc"].append(acc)
        result["test_results"]["itteration"]["acc@5"].append(acc_5)
        result["test_results"]["itteration"]["f1"].append(f1)
        cm_list.append(cm)
        result["test_results"]["itteration"]["cm"].append(cm.tolist())
        result["test_results"]["itteration"]["reports"].append(report)
        result["test_results"]["itteration"]["histry"].append(histry)
    
    cm1 = np.sum(cm_list,axis=0) / len(cm_list)
    sum_axis = cm_list[0].sum(axis=1)
    cm2 = cm1 / sum_axis[:, None]

    classes = [str(label) for label in np.unique(y_test)]
    average_metrics = average_metrics_for_reports(result["test_results"]["itteration"]["reports"], classes)

    result["test_results"]["result"]["cm_mean"]  = cm2.tolist()
    result["test_results"]["result"]["acc_mean"] = np.mean(result["test_results"]["itteration"]["acc"])
    result["test_results"]["result"]["acc_std"] = np.std(result["test_results"]["itteration"]["acc"])
    result["test_results"]["result"]["acc@5_mean"] = np.mean(result["test_results"]["itteration"]["acc@5"])
    result["test_results"]["result"]["acc@5_std"] = np.std(result["test_results"]["itteration"]["acc@5"])
    result["test_results"]["result"]["f1_mean"] = np.mean(result["test_results"]["itteration"]["f1"])
    result["test_results"]["result"]["f1_std"] = np.std(result["test_results"]["itteration"]["f1"])

    result["test_results"]["result"]["reports_mean"] = average_metrics 


    save_confusion_matrix(cm=cm2,
                          model=model_name,
                          head='mlp',
                          frames=frames,
                          finetuned=finetuned,
                          dataset= dataset,
                          labels=encoder.classes_,
                          path="results/confusion_matrix")
    
    print(f"[Test results] Acc: {result['test_results']['result']['acc_mean']} +- {result['test_results']['result']['acc_std']}")
    print(f"[Test results] Acc@5: {result['test_results']['result']['acc@5_mean']} +- {result['test_results']['result']['acc@5_std']}")
    print(f"[Test results] F1: {result['test_results']['result']['f1_mean']} +- {result['test_results']['result']['f1_std']}")

    append_metrics_to_csv(file_path="results/model_metric/model_metric.csv",
                          model=model_name,
                          head='mlp',
                          frames=frames,
                          finetuned=finetuned,
                          dataset=dataset,
                          acc=result["test_results"]["result"]["acc_mean"],
                          acc_std=result["test_results"]["result"]["acc_std"],
                          acc_5=result["test_results"]["result"]["acc@5_mean"],
                          acc_5_std=result["test_results"]["result"]["acc@5_std"],
                          f1=result["test_results"]["result"]["f1_mean"],
                          f1_std=result["test_results"]["result"]["f1_std"])

    return result