import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix, classification_report
from utils.utils import save_confusion_matrix, average_metrics_for_reports, append_metrics_to_csv
import warnings
warnings.filterwarnings('always')


def default_svm_params():
    svm = SVC()
    result = {"default_params" : svm.get_params()}
    return result


def search_kernel(X_train, 
                  X_val, 
                  y_val, 
                  y_train, 
                  iterration: int = 5, 
                  print_result: bool = False):
    
    result = {
        "kernel" : {
            "itteration" : {
                "param" : [],
                "acc" : [],
                "f1" : [] 
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

    for i in range(iterration):
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            
            svm = SVC(kernel=kernel)
            svm.fit(X_train,y_train)
            y_pred = svm.predict(X_val)

            acc = accuracy_score(y_val,y_pred)
            f1 = f1_score(y_val,y_pred,average='macro')
            
            if print_result:
                print(f"KERNEL: {kernel}, Acc: {acc}, F1: {f1}")
            
            result["kernel"]["itteration"]["param"].append(kernel)
            result["kernel"]["itteration"]["acc"].append(acc)
            result["kernel"]["itteration"]["f1"].append(f1)
    
    tmp = pd.DataFrame(result["kernel"]["itteration"])

    for param in tmp.param.unique():
        result["kernel"]["result_mean"]['param'].append(param)
        result["kernel"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["kernel"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["kernel"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["kernel"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["kernel"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["kernel"]["best_result"]['best_param'] = best_param
    result["kernel"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["kernel"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["kernel"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["kernel"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param kernel]: {best_param}, Acc: {result['kernel']['best_result']['acc_mean']} +- {result['kernel']['best_result']['acc_std']}, F1: {result['kernel']['best_result']['f1_mean']} +- {result['kernel']['best_result']['f1_std']}")

    return best_param, result


def search_dfc(X_train, 
               X_val,
               y_val, 
               y_train,
               kernel: str = "linear",
               iterration: int = 5,
               print_result: bool = False):
    
    result = {
        "dfc" : {
            "itteration" : {
                "param" : [],
                "acc" : [],
                "f1" : [] 
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

    for i in range(iterration):
        for dfc in ["ovr", "ovo"]:
            
            svm = SVC(kernel=kernel,
                      decision_function_shape=dfc)
            svm.fit(X_train,y_train)
            y_pred = svm.predict(X_val)

            acc = accuracy_score(y_val,y_pred)
            f1 = f1_score(y_val,y_pred,average='macro')
            
            if print_result:
                print(f"DFC: {dfc}, Acc: {acc}, F1: {f1}")
            
            result["dfc"]["itteration"]["param"].append(dfc)
            result["dfc"]["itteration"]["acc"].append(acc)
            result["dfc"]["itteration"]["f1"].append(f1)
    
    tmp = pd.DataFrame(result["dfc"]["itteration"])

    for param in tmp.param.unique():
        result["dfc"]["result_mean"]['param'].append(param)
        result["dfc"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["dfc"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["dfc"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["dfc"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["dfc"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["dfc"]["best_result"]['best_param'] = best_param
    result["dfc"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["dfc"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["dfc"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["dfc"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param dfc]: {best_param}, Acc: {result['dfc']['best_result']['acc_mean']} +- {result['dfc']['best_result']['acc_std']}, F1: {result['dfc']['best_result']['f1_mean']} +- {result['dfc']['best_result']['f1_std']}")

    return best_param, result

def search_c(X_train, 
             X_val,
             y_val, 
             y_train,
             kernel: str = "linear",
             dfc: str = "ovr",
             iterration: int = 5,
             print_result: bool = False):
    
    result = {
        "C" : {
            "itteration" : {
                "param" : [],
                "acc" : [],
                "f1" : [] 
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

    for i in range(iterration):
        for c in [item / 100 for item in range(20,710,20)]:
            
            svm = SVC(kernel=kernel,
                      decision_function_shape=dfc,
                      C=c)
            svm.fit(X_train,y_train)
            y_pred = svm.predict(X_val)

            acc = accuracy_score(y_val,y_pred)
            f1 = f1_score(y_val,y_pred,average='macro')
            
            if print_result:
                print(f"C: {c}, Acc: {acc}, F1: {f1}")
            
            result["C"]["itteration"]["param"].append(c)
            result["C"]["itteration"]["acc"].append(acc)
            result["C"]["itteration"]["f1"].append(f1)
    
    tmp = pd.DataFrame(result["C"]["itteration"])

    for param in tmp.param.unique():
        result["C"]["result_mean"]['param'].append(param)
        result["C"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["C"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["C"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["C"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["C"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["C"]["best_result"]['best_param'] = best_param
    result["C"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["C"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["C"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["C"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param C]: {best_param}, Acc: {result['C']['best_result']['acc_mean']} +- {result['C']['best_result']['acc_std']}, F1: {result['C']['best_result']['f1_mean']} +- {result['C']['best_result']['f1_std']}")

    return best_param, result

def test_best_param_svm(X_train,
                    X_test,
                    y_train, 
                    y_test,
                    encoder,
                    model,
                    frames,
                    finetuned,
                    dataset,
                    kernel: str = "linear",
                    dfc: str = "ovr",
                    c: float = 1.0,
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
        svm = SVC(kernel=kernel,
                C=c,
                decision_function_shape=dfc,
                probability=True)
        svm.fit(X_train,y_train)

        y_pred_test = svm.predict(X_test)
        y_pred_prob_test = svm.predict_proba(X_test)

        result["test_results"]["itteration"]["acc"].append(accuracy_score(y_test,y_pred_test))
        result["test_results"]["itteration"]["acc@5"].append(top_k_accuracy_score(y_test,y_pred_prob_test,k=5,labels=np.unique(y_train)))
        result["test_results"]["itteration"]["f1"].append(f1_score(y_test,y_pred_test,average='macro'))
        cm = confusion_matrix(y_test,y_pred_test)
        cm_list.append(cm)
        result["test_results"]["itteration"]["cm"].append(cm.tolist())
        cr_dict = classification_report(y_test,y_pred_test,output_dict=True)
        result["test_results"]["itteration"]["reports"].append(cr_dict)
    
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
                          model=model,
                          head='svm',
                          frames=frames,
                          finetuned=finetuned,
                          dataset= dataset,
                          labels=encoder.classes_,
                          path="results/confusion_matrix")
    
    print(f"[Test results] Acc: {result['test_results']['result']['acc_mean']} +- {result['test_results']['result']['acc_std']}")
    print(f"[Test results] Acc@5: {result['test_results']['result']['acc@5_mean']} +- {result['test_results']['result']['acc@5_std']}")
    print(f"[Test results] F1: {result['test_results']['result']['f1_mean']} +- {result['test_results']['result']['f1_std']}")

    append_metrics_to_csv(file_path="results/model_metric/model_metric.csv",
                          model=model,
                          head='svm',
                          frames=frames,
                          finetuned=finetuned,
                          dataset= dataset,
                          acc=result["test_results"]["result"]["acc_mean"],
                          acc_std=result["test_results"]["result"]["acc_std"],
                          acc_5=result["test_results"]["result"]["acc@5_mean"],
                          acc_5_std=result["test_results"]["result"]["acc@5_std"],
                          f1=result["test_results"]["result"]["f1_mean"],
                          f1_std=result["test_results"]["result"]["f1_std"])

    return result
    






