# Create dataset and save as .pt file

import os
import random

from load_griddata import (
    load_grid,
    save_torch_dataset,
)
from torch_geometric.loader import DataLoader

random.seed(2022)

datasets = {
    "aalbuhn": "aalbuhn",
    # "aalbuhn_small_changes": "aalbuhn_small_changes",
    # "tmdl": "tmdl_data",
    # "ap": "ap_data",
    # "arnhem": "arnhem_data",
}

val_fold = 0
normalise_data = False  # False
topology_changes = False  # True
undirected = True

# set parameters ----------
for dataset_explore in datasets:
    if dataset_explore == "arnhem":
        samples_fold = 200
    else:
        samples_fold = 10  # 2000

    print(f"samples per fold: {samples_fold} \n")

    # load data (griddata) ----------
    dataset = load_grid(
        dataset_explore,
        val_fold=val_fold,
        samples_per_fold=samples_fold,
        small_part=False,
        normalise=normalise_data,
        different_topo=topology_changes,
        undirected=undirected,
    )

    # save data (griddata) ----------
    # dataset = save_torch_dataset(
    #    dataset, dataset_explore, samples_fold, topo=topology_changes, undir=undirected, norm=normalise_data,
    # )

    # """
    print("---> create data with data loader ")
    train_loader = DataLoader(dataset.train_graphs, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset.val_graphs, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset.test_graphs, batch_size=1, shuffle=True)

    print("---> data in the train_loader ")
    for data in train_loader:
        print(
            f"Number of graphs: {data.num_graphs} \n"
            f"Number of nodes: {data.num_nodes} \n"
            f"Number of node features: {data.num_node_features}  \n"
            f"Number of edges: {data.num_edges}  \n"
            f"Number of edge features: {data.num_edge_features}  \n"
            f"Average node degree: {data.num_edges / data.num_nodes:.2f}  \n"
            f"Has isolated nodes: {data.has_isolated_nodes()}  \n"
            f"Has self-loops: {data.has_self_loops()} \n"
            f"Is undirected: {data.is_undirected()} \n"
        )
        print(data)
        break
    # """
