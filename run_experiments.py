import numpy as np
import json
import argparse
import torch

from utils.utils import str_to_bool, save_metadata, create_dataset, start_experiment, end_experiment
from utils.svm import default_svm_params, search_kernel, search_dfc, search_c, test_best_param_svm
from utils.random_forest import default_rf_params, search_criterion, search_n_estimators, search_min_samples_leaf, search_bootstrap, test_best_param_rf
from utils.mlp import default_mlp_params, search_batch_size, search_lr, search_dropout, search_epochs, test_best_param_mlp

from sklearn.preprocessing import LabelEncoder

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiments for research")

    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-m",
        "--model",
        help="Type of video model",
        choices=["mvitv2","uniformerv2","videomaev2"],
        required=True,
    )
    required.add_argument(
        "-fd",
        "--finetuned",
        help="Whether the model is finetuned",
        type=str_to_bool,
        required=True,
    )
    required.add_argument(
        "-hd",
        "--cls_head",
        help="Type of classifier head to use in the model",
        choices=["randomforest","svm","mlp","all"],
        required=True,
    )
    required.add_argument(
        "-d",
        "--dataset",
        help="Type of dataset",
        choices=["basketball","aerobic_gymnastics","diving","football","volleyball"],
        required=True,
    )
    required.add_argument(
        "-f",
        "--frame",
        help="Number of frames for a single video",
        type=int,
        choices=[8,16,32],
        default=8,
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    path_dir = "datasets_numpy"

    if args.finetuned:
        X_train = np.load(f"{path_dir}/video_feature_extraction/vfe_finetuned_{args.model}_frame{args.frame}_{args.dataset}_train.npy")
        X_val = np.load(f"{path_dir}/video_feature_extraction/vfe_finetuned_{args.model}_frame{args.frame}_{args.dataset}_val.npy")
        X_test = np.load(f"{path_dir}/video_feature_extraction/vfe_finetuned_{args.model}_frame{args.frame}_{args.dataset}_test.npy")
    else:
        X_train = np.load(f"{path_dir}/video_feature_extraction/vfe_{args.model}_frame{args.frame}_{args.dataset}_train.npy")
        X_val = np.load(f"{path_dir}/video_feature_extraction/vfe_{args.model}_frame{args.frame}_{args.dataset}_val.npy")
        X_test = np.load(f"{path_dir}/video_feature_extraction/vfe_{args.model}_frame{args.frame}_{args.dataset}_test.npy")
    
    with open(f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_train_metadata.json", 'r') as f:
        y_train_full = json.load(f)
    with open(f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_val_metadata.json", 'r') as f:
        y_val_full = json.load(f)
    with open(f"{path_dir}/frame_{args.frame}/frame{args.frame}_{args.dataset}_test_metadata.json", 'r') as f:
        y_test_full = json.load(f)

    encoder = LabelEncoder()
    encoder.fit(y_train_full['label'])

    y_train = encoder.transform(y_train_full['label'])
    y_val = encoder.transform(y_val_full['label'])
    y_test = encoder.transform(y_test_full['label'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(10*'#')
    print(f"Selected model: {args.model}")
    print(f"Dataset: {args.dataset} Finetuned: {args.finetuned} [Frames: {args.frame}]")
    print(f"Device: {device}")
    print(10*'#')

    start_time = start_experiment()

    if (args.cls_head == 'randomforest') | (args.cls_head == 'all'):
        print(10*'#')
        print(f"Testing cls_head: Random Forest")
        print()

        result0 = {
            "model" : args.model,
            "finetuned" : args.finetuned,
            "frames" : args.frame,
            "head" : 'randomforest',
            "dataset" : args.dataset
        }

        result1 = default_rf_params()
        criterion, result2 = search_criterion(X_train=X_train,
                                              X_val=X_val,
                                              y_val=y_val,
                                              y_train=y_train,
                                              iterration=3)
        n_estimators, result3 = search_n_estimators(X_train=X_train,
                                                    X_val=X_val,
                                                    y_val=y_val,
                                                    y_train=y_train,
                                                    criterion=criterion,
                                                    iterration=3)
        min_samples_leaf, result4 = search_min_samples_leaf(X_train=X_train,
                                                            X_val=X_val,
                                                            y_val=y_val,
                                                            y_train=y_train,
                                                            criterion=criterion,
                                                            n_estimator=n_estimators,
                                                            iterration=3)
        bootstrap, result5 = search_bootstrap(X_train=X_train,
                                              X_val=X_val,
                                              y_val=y_val,
                                              y_train=y_train,
                                              criterion=criterion,
                                              n_estimator=n_estimators,
                                              min_samples_leaf=min_samples_leaf,
                                              iterration=3)
        result6 = test_best_param_rf(X_train=X_train,
                                     X_test=X_test,
                                     y_train=y_train,
                                     y_test=y_test,
                                     encoder=encoder,
                                     model = args.model,
                                     frames= args.frame,
                                     finetuned= args.finetuned,
                                     dataset= args.dataset,
                                     criterion=criterion,
                                     n_estimator=n_estimators,
                                     min_samples_leaf=min_samples_leaf,
                                     bootstrap=bool(bootstrap),
                                     iterration=3)
        
        result = {**result0, **result1, **result2, **result3, **result4, **result5, **result6}

        save_metadata(model=args.model,
                      dataset=args.dataset,
                      frame=args.frame,
                      head='randomforest',
                      finetuned=args.finetuned,
                      result=result)

    if (args.cls_head == 'svm') | (args.cls_head == 'all'):
        print(10*'#')
        print(f"Testing cls_head: SVM")
        print()

        result0 = {
            "model" : args.model,
            "finetuned" : args.finetuned,
            "frames" : args.frame,
            "head" : 'svm',
            "dataset" : args.dataset
        }

        result1 = default_svm_params()
        kernel, result2 = search_kernel(X_train = X_train,
                                        X_val = X_val,
                                        y_val = y_val,
                                        y_train = y_train,
                                        iterration = 3)
        dfc, result3 = search_dfc(X_train = X_train, 
                                  X_val = X_val, 
                                  y_val = y_val, 
                                  y_train = y_train, 
                                  kernel = kernel, 
                                  iterration = 3)
        c, result4 = search_c(X_train = X_train, 
                              X_val = X_val, 
                              y_val = y_val, 
                              y_train = y_train, 
                              kernel = kernel, 
                              dfc = dfc, 
                              iterration = 3)
        result5 = test_best_param_svm(X_train = X_train, 
                                      X_test = X_test, 
                                      y_train = y_train, 
                                      y_test = y_test, 
                                      encoder = encoder,
                                      model = args.model,
                                      frames= args.frame,
                                      finetuned= args.finetuned,
                                      dataset= args.dataset,
                                      kernel = kernel, 
                                      dfc = dfc,
                                      c = c,
                                      iterration=3)
        result = {**result0, **result1, **result2, **result3, **result4, **result5}

        save_metadata(model=args.model,
                      dataset=args.dataset,
                      frame=args.frame,
                      head='svm',
                      finetuned=args.finetuned,
                      result=result)
        
    if (args.cls_head == 'mlp') | (args.cls_head == 'all'):

        input_dim=X_train.shape[1]
        class_size=len(np.unique(y_train))

        X_tensor, y_tensor, dataset = create_dataset(X_train, X_val, X_test, y_train, y_val, y_test)
    
        print(10*'#')
        print(f"Testing cls_head: MLP")
        print()

        result0 = {
            "model" : args.model,
            "finetuned" : args.finetuned,
            "frames" : args.frame,
            "head" : 'mlp',
            "dataset" : args.dataset
        }

        result1 = default_mlp_params()

        batch_size, result2 = search_batch_size(dataset_train=dataset[0],
                                                dataset_val=dataset[1],
                                                device=device,
                                                input_dim=input_dim,
                                                class_size=class_size,
                                                iterration=3)
        
        lr, result3 = search_lr(dataset_train=dataset[0],
                                dataset_val=dataset[1],
                                device=device,
                                input_dim=input_dim,
                                class_size=class_size,
                                batch_size=batch_size,
                                iterration=3)
        
        dropout, result4 = search_dropout(dataset_train=dataset[0],
                                            dataset_val=dataset[1],
                                            device=device,
                                            input_dim=input_dim,
                                            class_size=class_size,
                                            batch_size=batch_size,
                                            lr=lr,
                                            iterration=3)
        
        epochs, result5 = search_epochs(dataset_train=dataset[0],
                                        dataset_val=dataset[1],
                                        device=device,
                                        input_dim=input_dim,
                                        class_size=class_size,
                                        batch_size=batch_size,
                                        lr=lr,
                                        dropout_ratio=dropout,
                                        iterration=3)
        
        result6 = test_best_param_mlp(dataset_train=dataset[0],
                                        dataset_val=dataset[1],
                                        X_test_tensor=X_tensor[2],
                                        y_test_tensor=y_tensor[2],
                                        y_train_tensor=y_tensor[0],
                                        y_test=y_test,
                                        device=device,
                                        input_dim=input_dim,
                                        class_size=class_size,
                                        model_name=args.model,
                                        frames=args.frame,
                                        finetuned=args.finetuned,
                                        dataset=args.dataset,
                                        batch_size=batch_size,
                                        lr=lr,
                                        encoder=encoder,
                                        dropout_ratio=dropout,
                                        epochs=epochs,
                                        iterration=3)


        result = {**result0, **result1, **result2, **result3, **result4, **result5, **result6}

        save_metadata(model=args.model,
                      dataset=args.dataset,
                      frame=args.frame,
                      head='mlp',
                      finetuned=args.finetuned,
                      result=result)
    
    end_experiment(start_time=start_time,
                   model=args.model,
                   frames=args.frame,
                   head=args.cls_head,
                   finetuned=args.finetuned,
                   dataset=args.dataset,
                   file_path='results/logs/experiment_logs.csv')

if __name__ == "__main__":
    main()


