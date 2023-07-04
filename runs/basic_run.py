import os
import random
import torch
import torch.nn.functional as F

from GIN_model import GIN
from load_griddata import load_dataloader, load_torch_dataset, load_multiple_grid, load_dataloader_sampler
from calibrate_model import train_model, test_model, test_model_member, AUC_test
from torch.utils.data import SubsetRandomSampler
from functorch import combine_state_for_ensemble
import sys

os.chdir(os.getcwd())# + "/GINenergygrids")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

torch.manual_seed(2023)
print_log = True

merged_dataset = True #False (single)
normalise_features = False #True
topo_changes = True
undirected = True #True

calibrate_temperature = False #False (already optimised)

edge_classification = False

deep_ensemble = True
bootstrap = True # if false, a different dataset is used for each ensemble member (extra for-loop)
num_models = 10 #15 #amount of ensemble members, and data locations

save_plots = True
logging = "None"  # or wandb
txt_name = "ensemble_run_all_location_tscaled_seed"
print(txt_name)

# set parameters ----------
if merged_dataset:
    dataset_explore = [
        "aalbuhn_small_changes", #aalbuhn
        "tmdl",#"ap",
        "arnhem",
        "ap",#"tmdl",
    ]  # last one [-1] is the test data
    samples_fold = [2000, 2000, 200, 2000]  # TODO: make stand alone,
    print(
        f"Merging datasets [{dataset_explore[0], dataset_explore[1], dataset_explore[2]}] for training and using [{dataset_explore[3]}] for testing"
    )
else:
    print("Using a single dataset")
    dataset_explore = ["aalbuhn_small_changes"]#aalbuhn
    samples_fold = 2000

base_config = {
    "normalise_data": normalise_features,
    "topo_changes": topo_changes,
    "undirected": undirected,
    "edge_classification": edge_classification,
    "deep_ensemble": deep_ensemble,
    "bootstrap_de": bootstrap,
    "shuffle_data": True,
    "val_fold": 0,
    "edge_features": True,
    "linear_learn": False,
    "hidden_mlp": 16,
    "out_global": 1,
    "criterion": F.binary_cross_entropy,
    "calibrate_net": calibrate_temperature,
}

additional_config = {
    "dataset_explore": dataset_explore,
    "samples_fold": samples_fold,
    "batch_size": 32,  # 32, 64, [128]
    "activation_function_mlp": "LeakyReLU",  # or ReLU or LeakyReLU
    "activation_function_gin": "LeakyReLU",
    "aggregation_nodes_edges": "max",  # aggr=["mean", "add", "max"]
    "aggregation_global": "max",  # aggr=["mean", "add", "max"]
    "epochs": 125, # 100, #150,
    "num_layers": 15, #15,
    "dropout": 0.15, #0.15,
    "lr": 1e-6, #1e-5,
    "weightdecay": 0.01, #0.001
    "optimizer": "adam",
    "reduction_loss":"sum", #sum, mean
    "l1_weight": 0.01, #0.001
    "l2_weight": 0.01, #0.001
    "temperature_init":0.9, #1.0, None
}

additional_config.update(
    {"hidden_global": base_config["hidden_mlp"] * (additional_config.get("num_layers"))}
)

print(
    f"dataset explore: {additional_config['dataset_explore']} \n"
    f"edge classification: {base_config['edge_classification']} \n"
    f"samples per fold: {additional_config['samples_fold']} \n"
    f"batch size: {additional_config['batch_size']} \n"
    f"topology changes: {base_config['topo_changes']} \n"
    f"undirected edges: {base_config['undirected']} \n"
    f"normalised data: {base_config['normalise_data']} \n"
    f"linear learning: {base_config['linear_learn']} \n"
    f"deep ensembles: {base_config['deep_ensemble']} \n"
    f"deep ensembles with bootstrap: {base_config['bootstrap_de']} \n"
    f"calibrate temperature: {base_config['calibrate_net']} \n"
    f"init temperature: {additional_config['temperature_init']} \n"
    f"---------- \n"
    f"activation function gin: {additional_config['activation_function_gin']} \n"
    f"activation function mlp: {additional_config['activation_function_mlp']} \n"
    f"aggregation function nodes/edges: {additional_config['aggregation_nodes_edges']} \n"
    f"aggregation function global: {additional_config['aggregation_global']} \n"
    f"---------- \n"
    f"epochs: {additional_config['epochs']} \n"
    f"training with edge features: {base_config['edge_features']} \n"
    f"---------- \n"
    f"number of GIN layers: {additional_config['num_layers']} \n"
    f"dropout: {additional_config['dropout']} \n"
    f"learning rate: {additional_config['lr']} \n"
    f"optimser: {additional_config['optimizer']} \n"
    f"weight decay: {additional_config['weightdecay']} \n"
    f"criterion: {base_config['criterion']} \n"
    f"hidden global: {additional_config['hidden_global']} \n"
    f"reduction l1l2 loss: {additional_config['reduction_loss']} \n"
    f"l1 weight: {additional_config['l1_weight']} \n"
    f"l2 weight: {additional_config['l2_weight']} \n"
)


# load data (grid and dataloader) ----------
dataset = load_multiple_grid(
    additional_config["dataset_explore"],
    additional_config["samples_fold"],
    topo=base_config["topo_changes"],
    undir=base_config["undirected"],
    norm=base_config["normalise_data"],
    edge_class=base_config["edge_classification"]
)

train_loader, val_loader, test_loader = load_dataloader(
    dataset,
    batchsize=1,#additional_config["batch_size"],
    shuffle=base_config["shuffle_data"],
)

print(len(train_loader))

if base_config["bootstrap_de"]:
    num_train_examples = len(train_loader)
    num_val_examples = len(val_loader)
    
    print(f"subset samples: train ({num_train_examples}) and val ({num_val_examples}) examples")
    
    train_loader_sample = []
    val_loader_sample = []
    
    for i in range(num_models):
        train_loader_i, val_loader_i, _ = load_dataloader(
            dataset,
            batchsize=additional_config["batch_size"],
            shuffle=base_config["shuffle_data"],
        )
                
        train_loader_sample.append(train_loader_i)
        val_loader_sample.append(val_loader_i)

ensemble_seeds = [random.randint(2000, 2100) for _ in range(num_models)]
ensemble_seeds[0] = 2023

# load gnn ----------
print("---> creating the GNN with edge features ")
for loader in train_loader_sample:
    for data in train_loader:
        n_edge_feat = data.num_edge_features
        n_node_feat = data.num_node_features
        break

def create_ensemble_member (seed):
    torch.manual_seed(seed)
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
        edge_classification=base_config["edge_classification"]
    ).to(device)
    return model_gin_edges

model_gin_edges = create_ensemble_member(ensemble_seeds[0])
print(
    "amount of parameters in model: ",
    sum([param.nelement() for param in model_gin_edges.parameters()]),
)

torch.save(model_gin_edges.state_dict(), f"logs/models/model_gin_torch_{txt_name}.pth")

ensemble_models = [create_ensemble_member(ensemble_seeds[i]) for i in range(num_models)]

ensemble_params = [ensemble_models[i].parameters() for i in range(num_models)]

ensemble_BS = [0 for _ in range(num_models)]
ensemble_pred = [[] for _ in range(num_models)]

# train gnn ----------
for i in range(num_models):
    print(f"-- training ensemble {i}, with torch seed int({ensemble_seeds[i]})")
    torch.manual_seed(ensemble_seeds[i])
    model_gin_edges_i = ensemble_models[i]
    
    train_model(
        model_gin_edges_i,
        train_loader_sample[i],
        val_loader_sample[i],
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
        l2_weight=additional_config["l2_weight"],
        edge_classification=base_config["edge_classification"],
        temperature_init=additional_config["temperature_init"],
        calibrate_net=base_config["calibrate_net"],
    )
    ensemble_params[i] = model_gin_edges_i.parameters()
    
    total_loss, total_acc, BS_mean, pred_member = test_model_member(model_gin_edges_i, test_loader, txt_name, device, criterion=base_config["criterion"],
        test=True,
        reduction_loss=additional_config["reduction_loss"],
        l1_weight=additional_config["l1_weight"],
        l2_weight=additional_config["l2_weight"],
        temperature=additional_config["temperature_init"])
        
    i_auc = AUC_test(model_gin_edges_i, test_loader, device,
        criterion=base_config["criterion"],
        test=True,
        reduction_loss=additional_config["reduction_loss"],
        l1_weight=additional_config["l1_weight"],
        l2_weight=additional_config["l2_weight"],
        temperature=additional_config["temperature_init"]
    )
        
    ensemble_pred[i] = pred_member
    ensemble_BS[i] = BS_mean
    
    print(f"ensemble member {i}, BS score = {BS_mean}, AUC score = {i_auc}")
    
print(f"total variance of ensemble = {sum(ensemble_BS)/num_models}")

# test gnn on other locations ----------
