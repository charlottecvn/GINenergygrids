import os
import torch
import torch.nn.functional as F

from graphnetwork.GIN_model import GIN
from dataprocessing.load_griddata import (
    load_dataloader,
    load_multiple_grid,
)
from torchmetrics import MeanSquaredError

os.chdir(os.getcwd())  # + "/GINenergygrids")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

torch.manual_seed(2023)

print_log = True
merged_dataset = True
normalise_features = False
topo_changes = True
undirected = True
calibrate_temperature = False
deep_ensemble = True
bootstrap = True
num_ensembles = 1
save_plots = True

logging = "None"

if merged_dataset:
    dataset_explore = [
        "aalbuhn_small_changes",
        "tmdl",
        "arnhem",
        "ap",
    ]  # last one [-1] is the test data
    samples_fold = [2000, 2000, 200, 2000]
    print(
        f"Merging datasets [{dataset_explore[0], dataset_explore[1], dataset_explore[2]}] for training and using [{dataset_explore[3]}] for testing"
    )
else:
    print("Using a single dataset")
    dataset_explore = ["aalbuhn_small_changes"]
    samples_fold = 2000

base_config = {
    "normalise_data": normalise_features,
    "topo_changes": topo_changes,
    "undirected": undirected,
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
    "temperature_init": 0.9,
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
    f"deep ensembles: {base_config['deep_ensemble']} \n"
    f"deep ensembles with bootstrap: {base_config['bootstrap_de']} \n"
    f"calibrate temperature: {base_config['calibrate_net']} \n"
    f"init temperature: {additional_config['temperature_init']} \n"
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

# print(len(test_loader))

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
    "/logging/trained_model/model_gin_torch_ensemble_run_all_location_tscaled_seed.pth"
)
model_gin.load_state_dict(torch.load(model_path, map_location=device))
model_gin.eval()

"""
total_loss, total_acc, BS_mean, pred_member = test_model_member(model_gin, test_loader, "new", device, criterion=base_config["criterion"],
        test=True,
        reduction_loss=additional_config["reduction_loss"],
        l1_weight=additional_config["l1_weight"],
        l2_weight=additional_config["l2_weight"],
        temperature=additional_config["temperature_init"])

print(total_acc)
"""

# permutation feature importance
def score_model(model, data, data_x, data_z):
    logits, out_sigmoid = model(data_x, data.edge_index, data.batch, data_z)
    pred = logits  # out_sigmoid
    targets = data.y
    # metric_score = BinaryAUROC()#.to(device)
    metric_score = MeanSquaredError()  # MeanAbsoluteError()#MeanSquaredError()
    return metric_score(pred, targets)


score_orig = 0
data_orig = 0
for data in test_loader:
    if (
        data.edge_index[0].max() > len(data.batch) - 1
        or data.edge_index[1].max() > len(data.batch) - 1
    ):  # not test and
        pass
    else:
        score_ = score_model(model_gin, data, data.x, data.edge_attr)
        score_orig += score_
        data_orig += 1
score_orig = score_orig / data_orig
# print(f'score original (MSE): {score_orig}')

feat_importance = torch.empty(12, dtype=torch.float)

"""
node_feat_names = ['power consumption', 'init_U_MSR', 'closed_U_MSR', 'degree']
num_node_feat=4 
for i in range(0,num_node_feat):
    score_i = 0
    perm_d = 0
    count_pass = 0
    for d in test_loader:
        if (d.edge_index[0].max() > len(d.batch)-1 or d.edge_index[1].max() > len(d.batch)-1): #not test and
            pass
            count_pass += 1
        else: 
            #random permute (shuffle) column i of dataset d
            t = d.x.T[i]
            idx = torch.randperm(t.shape[0])
            t = t[idx].view(t.size())
            if torch.all(t.eq(d.x.T[i])):
                print("true t:", torch.all(t.eq(d.x.T[i])))
            d_new_x = d.x
            d_new_x[:, i] = t
            #compute score
            score_i_d = score_model(model_gin, d, d_new_x, d.edge_attr)
            score_i+=score_i_d
            perm_d+=1
           
    # importance column i 
    perm_imp_i = score_orig - score_i/perm_d#len(test_loader)
    feat_importance[i] = perm_imp_i
    print(f'importance node feat {node_feat_names[i]}({i}): {perm_imp_i}')

"""

edge_feat_names = [
    "impedance",
    "reactance",
    "I_NOM (nominal current)",
    "to_netopening",
    "init_I_cable",
    "closed_I_cable",
    "init_I_cable_/I_NOM",
    "closed_I_cable_/I_NOM",
]
num_edge_feat = 8
for i in range(0, num_edge_feat):
    score_i = 0
    perm_d = 0
    count_pass = 0
    for d in test_loader:
        if (
            d.edge_index[0].max() > len(d.batch) - 1
            or d.edge_index[1].max() > len(d.batch) - 1
        ):  # not test and
            pass
            count_pass += 1
        else:
            # random permute (shuffle) column i of dataset d
            t = d.edge_attr.T[i]
            idx = torch.randperm(t.shape[0])
            t = t[idx].view(t.size())
            if torch.all(t.eq(d.edge_attr.T[i])):
                print("true t:", torch.all(t.eq(d.edge_attr.T[i])))
            d_new_edge_attr = d.edge_attr
            d_new_edge_attr[:, i] = t
            # compute score
            score_i_d = score_model(model_gin, d, d.x, d_new_edge_attr)
            score_i += score_i_d
            perm_d += 1

    # importance column i
    perm_imp_i = score_orig - score_i / perm_d  # len(test_loader)
    # feat_importance[i+4] = perm_imp_i
    print(f"importance edge feat {edge_feat_names[i]}({i}): {perm_imp_i}")

print(feat_importance)
feat_importance = feat_importance[0:8]

"""
feat_importance = (feat_importance-min(feat_importance))/(max(feat_importance)-min(feat_importance))
print(feat_importance)

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

def plot_feature_importance(importance,names):
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title('FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()


feat_names = ['power consumption', 'init_U_MSR', 'closed_U_MSR', 'degree','impedance', 'reactance', 'I_NOM (nominal current)', 'to_netopening', 'init_I_cable', 'closed_I_cable', 'init_I_cable_/I_NOM', 'closed_I_cable_/I_NOM']
feat_importance_norm = feat_importance
plot_feature_importance(feat_importance_norm, feat_names)
"""
