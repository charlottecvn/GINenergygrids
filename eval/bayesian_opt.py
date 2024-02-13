import os
import torch
import torch.nn.functional as F
import click
from pathlib import Path
import optuna
import sys 
sys.path.append('/ceph/knmimo/GNNs_UQ_charlotte/GINenergygrids/')
from graphnetwork.GIN_model import GIN
from experiments.training import train_model, test_model, AUC_test
from dataprocessing.load_griddata import load_dataloader, load_multiple_grid

@click.command()
@click.option(
    "--trials",
    type=int,
    default=10,
    help="number of trials",
)
@click.option(
    "--merged_dataset",
    type=bool,
    required=True,
    help="single or loo",
)
@click.option(
    "--data_order",
    type=(str, str, str, str),
    required=False,
    help="dataset order (loo), last one [-1] is the test data",
)
@click.option(
    "--txt_name",
    type=str,
    required=False,
    default="optimization_trial",
    help="txt name for outputs",
)
@click.option(
    "--txt_optuna",
    type=str,
    required=False,
    default="test_optuna",
    help="txt name for optuna dashboard",
)

def main(trials: int, 
        merged_dataset: bool, 
        data_order: (str,str,str,str), 
        txt_name: str,
        txt_optuna: str,
        ):
    
    def run_experiment(k, epochs, merged_dataset, data_order, txt_name, batch_size, hidden_mlp, aggregation_nodes_edges, aggregation_global, activation_function_mlp, activation_function_gin, num_layers, dropout, lr, temp_init):
        torch.cuda.empty_cache()
        os.chdir('/ceph/knmimo/GNNs_UQ_charlotte/GINenergygrids/')
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        
        torch.manual_seed(2023)

        print_log = False
        logging = "None"
        txt_name = str(txt_name)
        print(txt_name)
        
        if merged_dataset:
            print(data_order)
        else:
            print(data_order[0])
            
        if merged_dataset:
            samples_fold = [2000, 2000, 2000, 2000] 
            for i in range(len(data_order)):
                if data_order[i]=="location3":
                    samples_fold[i]=200# last one [-1] is the test data
            
            print(
                f"Merging datasets [{data_order[0], data_order[1], data_order[2]}] for training "
                f"and using [{data_order[3]}] for testing"
            )
        else:
            print("Using a single dataset")
            data_order = [data_order[0]] 
            samples_fold=2000
            if data_order[0]=="location3":
                samples_fold = 200

        base_config = {
            "normalise_data": False,
            "topo_changes": True,
            "undirected": True,
            "shuffle_data": True,
            "val_fold": 0,
            "edge_features": True,
            "linear_learn": False,
            "hidden_mlp": hidden_mlp,
            "out_global": 1,
            "criterion": F.binary_cross_entropy
        }

        additional_config = {
            "dataset_explore": data_order,
            "samples_fold": samples_fold,
            "batch_size": batch_size,
            "activation_function_mlp": activation_function_mlp,
            "activation_function_gin": activation_function_gin,
            "aggregation_nodes_edges": aggregation_nodes_edges,
            "aggregation_global": aggregation_global,
            "epochs": epochs,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weightdecay": 0.01,
            "optimizer": "adam",
            "reduction_loss": "sum",
            "l1_weight": 0.01,
            "l2_weight": 0.01,
            "temperature_init": temp_init,
        }
        
        additional_config.update(
            {"hidden_global": base_config["hidden_mlp"] * (additional_config.get("num_layers"))}
        )
        
        print(f"Running experiment with config: {additional_config}")
        
        dataset = load_multiple_grid(
            additional_config["dataset_explore"],
            additional_config["samples_fold"],
            topo=base_config["topo_changes"],
            undir=base_config["undirected"],
            norm=base_config["normalise_data"]
        )

        _, _, test_loader = load_dataloader(
            dataset,
            batchsize=1, 
            shuffle=base_config["shuffle_data"],
        )

        train_loader, val_loader, _ = load_dataloader(
            dataset,
            batchsize=additional_config["batch_size"],
            shuffle=base_config["shuffle_data"],
        )

        for data in train_loader:
            n_edge_feat = data.num_edge_features
            n_node_feat = data.num_node_features
            break
        
        model_gin_edges = GIN(
            in_channels_gin_x=n_node_feat,
            in_channels_gin_edge=n_edge_feat,
            hidden_channels_gin=base_config["hidden_mlp"],
            out_channels_gin=base_config["hidden_mlp"],
            hidden_channels_global=additional_config["hidden_global"],
            out_channels_global=base_config["out_global"],
            num_layers=additional_config["num_layers"],
            edge_features=base_config["edge_features"],
            dropout=additional_config["dropout"],
            linear_learn=base_config["linear_learn"],
            activation_function_mlp=additional_config["activation_function_mlp"],
            activation_function_gin=additional_config["activation_function_gin"],
            aggregation_nodes_edges=additional_config["aggregation_nodes_edges"],
            aggregation_global=additional_config["aggregation_global"],
            device=device, 
        ).to(device)
        
        train_model(
            model_gin_edges,
            train_loader,  
            val_loader,
            test_loader,
            device,
            criterion=base_config["criterion"],
            n_epochs=additional_config["epochs"],
            opt_lr=additional_config["lr"],
            weightdecay=additional_config["weightdecay"],
            optimizer_=additional_config["optimizer"],
            logging=logging,
            print_log=print_log,
            name_log=txt_name,
            reduction_loss=additional_config["reduction_loss"],
            l1_weight=additional_config["l1_weight"],
            l2_weight=additional_config["l2_weight"]
        )

        total_loss, total_acc = test_model(
            model_gin_edges,
            test_loader,
            device,
            criterion=base_config["criterion"],
            test=True,
            reduction_loss=additional_config["reduction_loss"],
            l1_weight=additional_config["l1_weight"],
            l2_weight=additional_config["l2_weight"]
        )
        
        return total_loss, total_acc

    def objective(trial, n_trial, merged_dataset, data_order, txt_name):
        k = n_trial 
        epochs = trial.suggest_int('epochs', 250, 4000)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3) 
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        dropout = trial.suggest_uniform('dropout', 0.01, 0.5)
        temp_init = trial.suggest_uniform('temp_init', 0.8, 1.1)
        num_layers = trial.suggest_int('num_layers', 5, 25)
        aggregation_nodes_edges = trial.suggest_categorical('aggregation_nodes_edges', ['max', 'mean', 'sum'])
        aggregation_global = trial.suggest_categorical('aggregation_global', ['max', 'mean', 'sum'])
        activation_function_mlp = trial.suggest_categorical('activation_function_mlp', ['LeakyReLU', 'ReLU', 'Tanh'])
        activation_function_gin = trial.suggest_categorical('activation_function_gin', ['LeakyReLU', 'ReLU', 'Tanh'])
        hidden_mlp = trial.suggest_categorical('hidden_mlp' , [16, 32, 64, 128])
        
        hyperparams = {
            'k': k,
            'epochs': epochs,
            'merged_dataset': merged_dataset,  # Example fixed value
            'data_order': data_order,  # Example fixed value
            'txt_name': txt_name,
            'batch_size': batch_size,
            'hidden_mlp': hidden_mlp,  
            'aggregation_nodes_edges': aggregation_nodes_edges, 
            'aggregation_global': aggregation_global, 
            'activation_function_mlp': activation_function_mlp,  
            'activation_function_gin': activation_function_gin,  
            'num_layers': num_layers,  
            'dropout': dropout,
            'lr': lr,
            'temp_init': temp_init,  
        }

        loss, accuracy = run_experiment(**hyperparams)
        print(f"Trial finished with test loss {loss} and test accuracy {accuracy}")
        return accuracy
    
    study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3", study_name=txt_optuna)
    study.optimize(lambda trial: objective(trial, trials, merged_dataset, data_order, txt_name), n_trials=trials)
    
    optuna.plot_intermediate_values(study)
    optuna.plot_parallel_coordinate(study)
    optuna.plot_contour(study)
    
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print(study.best_trial.number)
    print(study.best_value)

if __name__ == '__main__':
    print("start hyperparam optimisation")
    main()
