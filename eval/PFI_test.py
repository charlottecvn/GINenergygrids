import os
import torch.nn.functional as F

from graphnetwork.GIN_model import GIN
from dataprocessing.load_griddata import (
    load_dataloader,
    load_multiple_grid,
)
from experiments.PFI import *

os.chdir(os.getcwd())  # + "/GINenergygrids")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")
print("Using device:", device)

torch.manual_seed(2023)

print_log = True
merged_dataset = True
normalise_features = False
topo_changes = True
undirected = True
save_plots = True

logging = "None"

if merged_dataset:
    dataset_explore = [
        "location1_small_changes",
        "location2",
        "location4",
        "location3",
    ]  # last one [-1] is the test data
    samples_fold = [2000, 2000, 200, 2000]
    print(
        f"Merging datasets [{dataset_explore[0], dataset_explore[1], dataset_explore[2]}] for training and using [{dataset_explore[3]}] for testing"
    )
else:
    print("Using a single dataset")
    dataset_explore = ["location1_small_changes"]
    samples_fold = 2000

base_config = {
    "normalise_data": normalise_features,
    "topo_changes": topo_changes,
    "undirected": undirected,
    "shuffle_data": True,
    "val_fold": 0,
    "edge_features": True,
    "linear_learn": False,
    "hidden_mlp": 16,
    "out_global": 1,
    "criterion": F.binary_cross_entropy,
}

additional_config = {
    "dataset_explore": dataset_explore,
    "samples_fold": samples_fold,
    "batch_size": 32,
    "activation_function_mlp": "LeakyReLU",
    "activation_function_gin": "LeakyReLU",
    "aggregation_nodes_edges": "max",
    "aggregation_global": "max",
    "epochs": 125,
    "num_layers": 15,
    "dropout": 0.15,
    "lr": 1e-6,
    "weightdecay": 0.01,
    "optimizer": "adam",
    "reduction_loss": "sum",
    "l1_weight": 0.01,
    "l2_weight": 0.01,
}

additional_config.update(
    {"hidden_global": base_config["hidden_mlp"] * (additional_config.get("num_layers"))}
)

print(
    f"dataset explore: {additional_config['dataset_explore']} \n"
    f"samples per fold: {additional_config['samples_fold']} \n"
    f"batch size: {additional_config['batch_size']} \n"
    f"topology changes: {base_config['topo_changes']} \n"
    f"undirected edges: {base_config['undirected']} \n"
    f"normalised data: {base_config['normalise_data']} \n"
    f"linear learning: {base_config['linear_learn']} \n"
    f"activation function gin: {additional_config['activation_function_gin']} \n"
    f"activation function mlp: {additional_config['activation_function_mlp']} \n"
    f"aggregation function nodes/edges: {additional_config['aggregation_nodes_edges']} \n"
    f"aggregation function global: {additional_config['aggregation_global']} \n"
    f"epochs: {additional_config['epochs']} \n"
    f"training with edge features: {base_config['edge_features']} \n"
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

dataset = load_multiple_grid(
    additional_config["dataset_explore"],
    additional_config["samples_fold"],
    topo=base_config["topo_changes"],
    undir=base_config["undirected"],
    norm=base_config["normalise_data"],
)

train_loader, val_loader, test_loader = load_dataloader(
    dataset,
    batchsize=1,
    shuffle=base_config["shuffle_data"],
)

for loader in train_loader:
    for data in train_loader:
        n_edge_feat = data.num_edge_features
        n_node_feat = data.num_node_features
        break

model_gin = GIN(
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
).to(device)

model_path = (
    "/logs/trained_model/model_gin_torch_ensemble_run_all_location_tscaled_seed.pth"
)
model_gin.load_state_dict(torch.load(model_path, map_location=device))
model_gin.eval()

PFI_model(test_loader, model_gin)
