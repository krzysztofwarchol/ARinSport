import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix, classification_report
from utils.utils import save_confusion_matrix, average_metrics_for_reports, append_metrics_to_csv
import warnings
warnings.filterwarnings('always')


def default_rf_params():
    rf = RandomForestClassifier()
    result = {"default_params" : rf.get_params()}
    return result

def search_criterion(X_train,
                     X_val,
                     y_val,
                     y_train,
                     iterration: int = 5, 
                     print_result: bool = False):

    result = {
        "criterion" : {
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

    for _ in range(iterration):
        for criterion in ["gini","entropy", "log_loss"]:
            
            rf = RandomForestClassifier(criterion=criterion)
            rf.fit(X_train,y_train)
            y_pred = rf.predict(X_val)

            acc = accuracy_score(y_val,y_pred)
            f1 = f1_score(y_val,y_pred,average='macro')
            
            if print_result:
                print(f"CRITERION: {criterion}, Acc: {acc}, F1: {f1}")
            
            result["criterion"]["itteration"]["param"].append(criterion)
            result["criterion"]["itteration"]["acc"].append(acc)
            result["criterion"]["itteration"]["f1"].append(f1)
    
    tmp = pd.DataFrame(result["criterion"]["itteration"])

    for param in tmp.param.unique():
        result["criterion"]["result_mean"]['param'].append(param)
        result["criterion"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["criterion"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["criterion"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["criterion"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["criterion"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["criterion"]["best_result"]['best_param'] = best_param
    result["criterion"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["criterion"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["criterion"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["criterion"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param criterion]: {best_param}, Acc: {result['criterion']['best_result']['acc_mean']} +- {result['criterion']['best_result']['acc_std']}, F1: {result['criterion']['best_result']['f1_mean']} +- {result['criterion']['best_result']['f1_std']}")

    return best_param, result


def search_n_estimators(X_train, 
                        X_val,y_val, 
                        y_train,
                        criterion: str = "gini",
                        iterration: int = 5,
                        print_result: bool = False):
    
    result = {
        "n_estimator" : {
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

    for _ in range(iterration):
        for n_estimator in [item for item in range(25,310,25)]:
            
            rf = RandomForestClassifier(criterion=criterion,
                                        n_estimators=n_estimator)
            rf.fit(X_train,y_train)
            y_pred = rf.predict(X_val)

            acc = accuracy_score(y_val,y_pred)
            f1 = f1_score(y_val,y_pred,average='macro')
            
            if print_result:
                print(f"N_ESTIMATOR: {n_estimator}, Acc: {acc}, F1: {f1}")
            
            result["n_estimator"]["itteration"]["param"].append(n_estimator)
            result["n_estimator"]["itteration"]["acc"].append(acc)
            result["n_estimator"]["itteration"]["f1"].append(f1)
    
    tmp = pd.DataFrame(result["n_estimator"]["itteration"])

    for param in tmp.param.unique():
        result["n_estimator"]["result_mean"]['param'].append(int(param))
        result["n_estimator"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["n_estimator"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["n_estimator"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["n_estimator"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["n_estimator"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["n_estimator"]["best_result"]['best_param'] = int(best_param)
    result["n_estimator"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["n_estimator"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["n_estimator"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["n_estimator"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param n_estimator]: {best_param}, Acc: {result['n_estimator']['best_result']['acc_mean']} +- {result['n_estimator']['best_result']['acc_std']}, F1: {result['n_estimator']['best_result']['f1_mean']} +- {result['n_estimator']['best_result']['f1_std']}")

    return best_param, result


def search_min_samples_leaf(X_train, 
                            X_val,y_val, 
                            y_train, 
                            criterion: str = "gini", 
                            n_estimator: int = 100, 
                            iterration: int = 5,
                            print_result: bool = False):
    
    result = {
        "min_samples_leaf" : {
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

    for _ in range(iterration):
        for msl in [item for item in range(1,11)]:
            
            rf = RandomForestClassifier(criterion=criterion,
                                        n_estimators=n_estimator,
                                        min_samples_leaf=msl)
            rf.fit(X_train,y_train)
            y_pred = rf.predict(X_val)

            acc = accuracy_score(y_val,y_pred)
            f1 = f1_score(y_val,y_pred,average='macro')
            
            if print_result:
                print(f"MIN_SAMPLES_LEAF: {msl}, Acc: {acc}, F1: {f1}")
            
            result["min_samples_leaf"]["itteration"]["param"].append(msl)
            result["min_samples_leaf"]["itteration"]["acc"].append(acc)
            result["min_samples_leaf"]["itteration"]["f1"].append(f1)
    
    tmp = pd.DataFrame(result["min_samples_leaf"]["itteration"])

    for param in tmp.param.unique():
        result["min_samples_leaf"]["result_mean"]['param'].append(int(param))
        result["min_samples_leaf"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["min_samples_leaf"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["min_samples_leaf"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["min_samples_leaf"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["min_samples_leaf"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["min_samples_leaf"]["best_result"]['best_param'] = int(best_param)
    result["min_samples_leaf"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["min_samples_leaf"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["min_samples_leaf"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["min_samples_leaf"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param min_samples_leaf]: {best_param}, Acc: {result['min_samples_leaf']['best_result']['acc_mean']} +- {result['min_samples_leaf']['best_result']['acc_std']}, F1: {result['min_samples_leaf']['best_result']['f1_mean']} +- {result['min_samples_leaf']['best_result']['f1_std']}")

    return best_param, result


def search_bootstrap(X_train, 
                     X_val,
                     y_val, 
                     y_train, 
                     criterion: str = "gini", 
                     n_estimator: int = 100,
                     min_samples_leaf: int = 1, 
                     iterration: int = 5,
                     print_result: bool = False):
    
    result = {
        "bootstrap" : {
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

    for _ in range(iterration):
        for bp in [True,False]:
            
            rf = RandomForestClassifier(criterion=criterion,
                                        n_estimators=n_estimator,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=bp)
            rf.fit(X_train,y_train)
            y_pred = rf.predict(X_val)

            acc = accuracy_score(y_val,y_pred)
            f1 = f1_score(y_val,y_pred,average='macro')
            
            if print_result:
                print(f"BOOTSTRAP: {bp}, Acc: {acc}, F1: {f1}")
            
            result["bootstrap"]["itteration"]["param"].append(bp)
            result["bootstrap"]["itteration"]["acc"].append(acc)
            result["bootstrap"]["itteration"]["f1"].append(f1)
    
    tmp = pd.DataFrame(result["bootstrap"]["itteration"])

    for param in tmp.param.unique():
        result["bootstrap"]["result_mean"]['param'].append(int(param))
        result["bootstrap"]["result_mean"]['acc_mean'].append(tmp[tmp['param'] == param]['acc'].mean())
        result["bootstrap"]["result_mean"]['acc_std'].append(tmp[tmp['param'] == param]['acc'].std())
        result["bootstrap"]["result_mean"]['f1_mean'].append(tmp[tmp['param'] == param]['f1'].mean())
        result["bootstrap"]["result_mean"]['f1_std'].append(tmp[tmp['param'] == param]['f1'].std())

    tmp2 = pd.DataFrame(result["bootstrap"]["result_mean"]).sort_values(by=['acc_mean','acc_std','f1_mean','f1_std'],ascending=[False,True,False,True])

    best_param = tmp2.head(1).param.values[0]

    result["bootstrap"]["best_result"]['best_param'] = int(best_param)
    result["bootstrap"]["best_result"]['acc_mean'] = tmp2.head(1).acc_mean.values[0]
    result["bootstrap"]["best_result"]['acc_std'] = tmp2.head(1).acc_std.values[0]
    result["bootstrap"]["best_result"]['f1_mean'] = tmp2.head(1).f1_mean.values[0]
    result["bootstrap"]["best_result"]['f1_std'] = tmp2.head(1).f1_std.values[0]
    
    print(f"[Best_param bootstrap]: {best_param}, Acc: {result['bootstrap']['best_result']['acc_mean']} +- {result['bootstrap']['best_result']['acc_std']}, F1: {result['bootstrap']['best_result']['f1_mean']} +- {result['bootstrap']['best_result']['f1_std']}")

    return best_param, result


def test_best_param_rf(X_train,
                    X_test,
                    y_train, 
                    y_test,
                    encoder,
                    model,
                    frames,
                    finetuned,
                    dataset,
                    criterion: str = "gini", 
                    n_estimator: int = 100,
                    min_samples_leaf: int = 1,
                    bootstrap: bool = True,
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

    for _ in range(iterration):
        rf = RandomForestClassifier(criterion=criterion,
                                    n_estimators=n_estimator,
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=bootstrap)
        rf.fit(X_train,y_train)

        y_pred_test = rf.predict(X_test)
        y_pred_prob_test = rf.predict_proba(X_test)

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
                          head='randomforest',
                          frames=frames,
                          finetuned=finetuned,
                          dataset=dataset,
                          labels=encoder.classes_,
                          path="results/confusion_matrix")
    
    print(f"[Test results] Acc: {result['test_results']['result']['acc_mean']} +- {result['test_results']['result']['acc_std']}")
    print(f"[Test results] Acc@5: {result['test_results']['result']['acc@5_mean']} +- {result['test_results']['result']['acc@5_std']}")
    print(f"[Test results] F1: {result['test_results']['result']['f1_mean']} +- {result['test_results']['result']['f1_std']}")

    append_metrics_to_csv(file_path="results/model_metric/model_metric.csv",
                          model=model,
                          head='randomforest',
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


