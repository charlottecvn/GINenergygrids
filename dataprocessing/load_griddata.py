# load the correct data into a DataLoader

import os

import torch
from torch_geometric.loader import DataLoader
from custompyg.pygenergydata import GridDataset

datasets = {
    "location1": "location1",
    "location1_small_changes": "location1_small_changes",
    "location2": "location2_data",
    "location3": "location3_data",
    "location4": "location4_data",
    "all": "all",
}

os.chdir(os.getcwd())  # + "/dataprocessing")


def load_grid(
    dataset_explore,
    val_fold=0,
    samples_per_fold=20,
    small_part=False,
    normalise=False,
    different_topo=False,
    undirected=False,
):
    directory = os.getcwd()

    # load data ----------
    print("---> start dataset loading ", dataset_explore)
    dataset = GridDataset(
        directory=directory,
        val_fold=val_fold,
        samples_per_fold=samples_per_fold,
        small_part=small_part,
        normalise=normalise,
        different_topo=different_topo,
        undirected=undirected,
    )

    """
    print("-- amount of (sub)graphs")
    print("train graphs: ", len(dataset.train_graphs))
    print("validation graphs: ", len(dataset.val_graphs))
    print("test graphs: ", len(dataset.test_graphs))
    """

    return dataset


def load_dataloader(dataset, batchsize=64, shuffle=True):
    # put into DataLoader ----------
    # print("---> create data with data loader ")
    train_loader = DataLoader(
        dataset.train_graphs, batch_size=batchsize, shuffle=shuffle
    )
    val_loader = DataLoader(dataset.val_graphs, batch_size=batchsize, shuffle=shuffle)
    test_loader = DataLoader(dataset.test_graphs, batch_size=batchsize, shuffle=shuffle)

    # print("---> data in the train_loader ")
    for data in train_loader:
        """
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
        """
        print(data)
        break

    return train_loader, val_loader, test_loader


def load_dataloader_sampler(dataset, sampler_train, sampler_val, batchsize=64):
    # put into DataLoader using a random subset sampler ----------
    # print("---> create data with data loader (random subset sampler)")
    train_loader = DataLoader(
        dataset.train_graphs, batch_size=batchsize, shuffle=False, sampler=sampler_train
    )
    val_loader = DataLoader(
        dataset.val_graphs, batch_size=batchsize, shuffle=False, sampler=sampler_val
    )
    test_loader = DataLoader(dataset.test_graphs, batch_size=batchsize, shuffle=True)

    return train_loader, val_loader, test_loader


def save_torch_dataset(
    dataset, dataset_explore, samples_fold, topo=False, undir=False, norm=False
):
    if topo and undir and norm:
        location_save = f"data_pt/grid_data_{dataset_explore}_topo_undirected_norm_{samples_fold}.pt"
    elif topo and undir:
        location_save = (
            f"data_pt/grid_data_{dataset_explore}_topo_undirected_{samples_fold}.pt"
        )
    elif topo and norm:
        location_save = (
            f"data_pt/grid_data_{dataset_explore}_topo_norm_{samples_fold}.pt"
        )
    elif norm and undir:
        location_save = (
            f"data_pt/grid_data_{dataset_explore}_undirected_norm_{samples_fold}.pt"
        )
    elif topo:
        location_save = f"data_pt/grid_data_{dataset_explore}_topo_{samples_fold}.pt"
    elif undir:
        location_save = (
            f"data_pt/grid_data_{dataset_explore}_undirected_{samples_fold}.pt"
        )
    elif norm:
        location_save = f"data_pt/grid_data_{dataset_explore}_norm_{samples_fold}.pt"
    else:
        location_save = f"data_pt/grid_data_{dataset_explore}_{samples_fold}.pt"

    torch.save(dataset, location_save)


def load_torch_dataset(
    dataset_explore,
    samples_fold,
    topo=False,
    undir=False,
    norm=False,
):
    # print("---> loading data from torch file ")
    # print("Topology changes: ", topo)
    # print("Undirected: ", undir)
    # print("Normalised: ", norm)

    if topo and undir and norm:
        location_save = f"data_pt/grid_data_{dataset_explore}_topo_undirected_norm_{samples_fold}.pt"
    elif topo and undir:
        location_save = (
            f"data_pt/grid_data_{dataset_explore}_topo_undirected_{samples_fold}.pt"
        )
    elif topo and norm:
        location_save = (
            f"data_pt/grid_data_{dataset_explore}_topo_norm_{samples_fold}.pt"
        )
    elif norm and undir:
        location_save = (
            f"data_pt/grid_data_{dataset_explore}_undirected_norm_{samples_fold}.pt"
        )
    elif topo:
        location_save = f"data_pt/grid_data_{dataset_explore}_topo_{samples_fold}.pt"
    elif undir:
        location_save = (
            f"data_pt/grid_data_{dataset_explore}_undirected_{samples_fold}.pt"
        )
    elif norm:
        location_save = f"data_pt/grid_data_{dataset_explore}_norm_{samples_fold}.pt"
    else:
        location_save = f"data_pt/grid_data_{dataset_explore}_{samples_fold}.pt"

    # print("test")
    # print(location_save)
    loaded_torch = torch.load(location_save)

    # """
    # print("-- amount of (sub)graphs")
    print("dataset explore: ", dataset_explore)
    print("train graphs: ", len(loaded_torch.train_graphs))
    print("validation graphs: ", len(loaded_torch.val_graphs))
    print("test graphs: ", len(loaded_torch.test_graphs))
    # """

    return loaded_torch


def concat_datagrid(data_1, data_2):
    concat_data = data_1  # custom_pyg_data.GridDataset.__init__()

    concat_data.train_graphs = torch.utils.data.ConcatDataset(
        [data_1.train_graphs, data_2.train_graphs]
    )
    concat_data.val_graphs = torch.utils.data.ConcatDataset(
        [data_1.val_graphs, data_2.val_graphs]
    )
    concat_data.train_targets = torch.utils.data.ConcatDataset(
        [data_1.train_targets, data_2.train_targets]
    )
    concat_data.val_targets = torch.utils.data.ConcatDataset(
        [data_1.val_targets, data_2.val_targets]
    )

    return concat_data


def load_multiple_grid(
    dataset_explore,
    samples_fold,
    norm=False,
    topo=False,
    undir=False,
):
    # print("---> loading multi data from torch file ")
    print("Topology changes: ", topo)
    print("Undirected: ", undir)
    print("Normalised: ", norm)

    if len(dataset_explore) > 1:
        dataset_1 = load_torch_dataset(
            dataset_explore[0],
            samples_fold[0],
            topo=topo,
            undir=undir,
            norm=norm,
        )
        dataset_2 = load_torch_dataset(
            dataset_explore[1],
            samples_fold[1],
            topo=topo,
            undir=undir,
            norm=norm,
        )
        dataset_3 = load_torch_dataset(
            dataset_explore[2],
            samples_fold[2],
            topo=topo,
            undir=undir,
            norm=norm,
        )
        dataset_4 = load_torch_dataset(
            dataset_explore[3],
            samples_fold[3],
            topo=topo,
            undir=undir,
            norm=norm,
        )

        concated_data = concat_datagrid(dataset_1, dataset_2)
        data_all = concat_datagrid(concated_data, dataset_3)

        data_all.test_graphs = dataset_4.train_graphs
        data_all.test_targets = dataset_4.train_targets
    else:
        data_all = load_torch_dataset(
            dataset_explore[0],
            samples_fold,
            topo=topo,
            undir=undir,
            norm=norm,
        )
    return data_all
