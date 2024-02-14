import os
import sys
import torch
import torch.nn.functional as F
import click
from pathlib import Path

sys.path.append("/ceph/knmimo/GNNs_UQ_charlotte/GINenergygrids/")

from graphnetwork.GIN_model import GIN
from experiments.training import train_model, test_model, AUC_test
from dataprocessing.load_griddata import (
    load_dataloader,
    load_multiple_grid,
)


@click.command()
@click.option(
    "--k",
    type=int,
    default=1,
    help="the value of kth experiment",
)
@click.option(
    "--epochs",
    type=int,
    required=True,
    help="number of epochs for training",
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
    help="txt name for outputs",
)
@click.option(
    "--batch_size",
    type=int,
    required=False,
    default=32,
    help="batch size",
)
@click.option(
    "--hidden_mlp",
    type=int,
    required=False,
    default=16,
    help="size hidden layers or MLP",
)
@click.option(
    "--aggregation_nodes_edges",
    type=str,
    required=False,
    default="max",
    help="aggregation nodes and edges",
)
@click.option(
    "--aggregation_global",
    type=str,
    required=False,
    default="max",
    help="aggregation global",
)
@click.option(
    "--activation_function_mlp",
    type=str,
    required=False,
    default="LeakyReLU",
    help="activation function mlp",
)
@click.option(
    "--activation_function_gin",
    type=str,
    required=False,
    default="LeakyReLU",
    help="activation function gin",
)
@click.option(
    "--num_layers",
    type=int,
    required=False,
    default=15,
    help="number of layers (related to k-hop neighbourhood)",
)
@click.option(
    "--dropout",
    type=float,
    required=False,
    default=0.15,
    help="dropout",
)
@click.option(
    "--lr",
    type=float,
    required=False,
    default=1e-6,
    help="learning rate",
)
@click.option(
    "--temp_init",
    type=float,
    required=False,
    default=0.9,
    help="temperature (calibration)",
)
# "weightdecay": 0.01, "optimizer": "adam", "reduction_loss": "sum",  "l1_weight": 0.01, "l2_weight": 0.01,  "temperature_init": 0.9,


def main(
    k: int,
    epochs: int,
    merged_dataset: bool,
    data_order: (str, str, str, str),
    txt_name: str,
    batch_size: int,
    hidden_mlp: int,
    aggregation_nodes_edges: str,
    aggregation_global: str,
    activation_function_mlp: str,
    activation_function_gin: str,
    num_layers: int,
    dropout: float,
    lr: float,
    temp_init: float,
):
    os.chdir("/ceph/knmimo/GNNs_UQ_charlotte/GINenergygrids/")
    print(os.getcwd())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(torch.version.cuda)
    print("Is CUDA enabled?", torch.cuda.is_available())

    n_epochs = epochs

    torch.manual_seed(2023)
    print_log = True

    normalise_features = False  # True
    topo_changes = True
    undirected = True  # True

    save_plots = True
    logging = "None"
    txt_name = str(txt_name)
    print(txt_name)

    if merged_dataset:
        print(data_order)
    else:
        print(data_order[0])

    # set parameters ----------
    if merged_dataset:
        samples_fold = [2000, 2000, 2000, 2000]
        for i in range(len(data_order)):
            if data_order[i] == "location3":
                samples_fold[i] = 200  # last one [-1] is the test data

        print(
            f"Merging datasets [{data_order[0], data_order[1], data_order[2]}] for training "
            f"and using [{data_order[3]}] for testing"
        )
    else:
        print("Using a single dataset")
        data_order = [data_order[0]]
        samples_fold = 2000
        if data_order[0] == "location3":
            samples_fold = 200

    base_config = {
        "normalise_data": normalise_features,
        "topo_changes": topo_changes,
        "undirected": undirected,
        "shuffle_data": True,
        "val_fold": 0,
        "edge_features": True,
        "linear_learn": False,
        "hidden_mlp": hidden_mlp,
        "out_global": 1,
        "criterion": F.binary_cross_entropy,
    }

    additional_config = {
        "dataset_explore": data_order,
        "samples_fold": samples_fold,
        "batch_size": batch_size,  # 32, 64, [128]
        "activation_function_mlp": activation_function_mlp,  # or ReLU or LeakyReLU
        "activation_function_gin": activation_function_gin,
        "aggregation_nodes_edges": aggregation_nodes_edges,  # aggr=["mean", "add", "max"]
        "aggregation_global": aggregation_global,  # aggr=["mean", "add", "max"]
        "epochs": n_epochs,  # 100, #150,
        "num_layers": num_layers,  # 15,
        "dropout": dropout,  # 0.15,
        "lr": lr,  # 1e-5,
        "weightdecay": 0.01,  # 0.001
        "optimizer": "adam",
        "reduction_loss": "sum",  # sum, mean
        "l1_weight": 0.01,  # 0.001
        "l2_weight": 0.01,  # 0.001
        "temperature_init": temp_init,  # 1.0, None
    }

    additional_config.update(
        {
            "hidden_global": base_config["hidden_mlp"]
            * (additional_config.get("num_layers"))
        }
    )

    print(
        f"dataset explore: {additional_config['dataset_explore']} \n"
        f"samples per fold: {additional_config['samples_fold']} \n"
        f"batch size: {additional_config['batch_size']} \n"
        f"topology changes: {base_config['topo_changes']} \n"
        f"undirected edges: {base_config['undirected']} \n"
        f"normalised data: {base_config['normalise_data']} \n"
        f"linear learning: {base_config['linear_learn']} \n"
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
    )

    _, _, test_loader = load_dataloader(
        dataset,
        batchsize=1,  # additional_config["batch_size"],
        shuffle=base_config["shuffle_data"],
    )

    train_loader, val_loader, _ = load_dataloader(
        dataset,
        batchsize=additional_config["batch_size"],
        shuffle=base_config["shuffle_data"],
    )

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    # load gnn ----------
    print("---> creating the GNN with edge features ")
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

    print(
        "amount of parameters in model: ",
        sum([param.nelement() for param in model_gin_edges.parameters()]),
    )

    torch.save(
        model_gin_edges.state_dict(), f"../logs/models/model_gin_torch_{txt_name}.pth"
    )

    # train gnn ----------
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
        l2_weight=additional_config["l2_weight"],
    )

    torch.save(
        model_gin_edges.state_dict(), f"../logs/models/model_gin_torch_{txt_name}.pth"
    )

    """
    total_loss, total_acc, pred_ = test_model(
        model_gin_edges,
        test_loader,
        txt_name,
        device,
        criterion=base_config["criterion"],
        test=True,
        reduction_loss=additional_config["reduction_loss"],
        l1_weight=additional_config["l1_weight"],
        l2_weight=additional_config["l2_weight"]
    )
    """

    total_loss, total_acc = test_model(
        model_gin_edges,
        test_loader,
        device,
        criterion=base_config["criterion"],
        test=True,
        reduction_loss=additional_config["reduction_loss"],
        l1_weight=additional_config["l1_weight"],
        l2_weight=additional_config["l2_weight"],
    )

    AUC_test(
        model_gin_edges,
        test_loader,
        device,
        # criterion=base_config["criterion"],
        test=True,
        # reduction_loss=additional_config["reduction_loss"],
        # l1_weight=additional_config["l1_weight"],
        # l2_weight=additional_config["l2_weight"]
    )

    return total_loss, total_acc


if __name__ == "__main__":
    main()
