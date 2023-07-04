# Data exploration for the grid data (Alliander)

import os

# from grid_data import GridDataset
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

datasets = {
    "aalbuhn": "aalbuhn",
    "aalbuhn_small_changes": "aalbuhn_small_changes",
    "tmdl": "tmdl_data",
    "ap": "ap_data",
    "arnhem": "arnhem_data",
}

for dataset_explore in datasets:
    print("--------> start data exploration on ", dataset_explore)

    directory = "example_data/" + datasets[dataset_explore]
    folders = os.listdir(directory)
    print("amount of folds: ", len(folders) - 3)

    # load data ----------
    """
    print("---> start data loading ", dataset_explore)
    dataset = GridDataset("example_data/"+datasets[dataset_explore], val_fold=0, samples_per_fold=2000, small_part=False)  # TODO: 2000
    
    print("-- amount of examples")
    num_train_examples = len(dataset.train_targets)
    print("train examples: ", num_train_examples)
    num_val_examples = len(dataset.val_targets)
    print("validation examples: ", num_val_examples)
    num_test_examples = len(dataset.test_targets)
    print("test examples: ", num_test_examples)
    
    print("-- amount of (sub)graphs")
    print("train graphs: ", len(dataset.train_graphs))
    print(dataset.train_graphs)
    print("validation graphs: ", len(dataset.val_graphs))
    print(dataset.val_graphs)
    print("test graphs: ", len(dataset.test_graphs))
    print(dataset.test_graphs)
    
    # put into DataLoader ----------
    train_loader = DataLoader([dataset.train_graphs, dataset.train_graphs], batch_size=64, shuffle=True)
    val_loader = DataLoader([dataset.val_graphs, dataset.val_graphs], batch_size=64, shuffle=True)
    test_loader = DataLoader([dataset.test_graphs, dataset.test_graphs], batch_size=64, shuffle=True)
    """

    """
    print('\nTrain loader:')
    for i, subgraph in enumerate(dataset.train_graphs):
        print(f' - Subgraph {i}: {subgraph}')
        #print("number of node features: ", subgraph.num_node_features)
        #print("number of classes: ", subgraph.num_classes)
    """

    # n-1 grids ----------
    print("---> start n-1 ", "")

    zero_counter = 0
    all_counter = 0
    targets = []
    for folder in folders:
        if not "fold_" in folder:
            continue
        for iteration in os.listdir(os.path.join(directory, folder)):
            unswitchables = len(
                pd.read_csv(
                    os.path.join(
                        directory, folder, iteration, "unswitchables_by_capacities.csv"
                    )
                )
            )
            if unswitchables == 0:
                targets.append(1.0)
                zero_counter += 1
            else:
                targets.append(0.0)
            all_counter += 1

    print("Percentage of grids that are n-1: ", zero_counter / all_counter)

    # grid and edge structure ----------
    print("---> grid and edge/vertices structure ", "")
    path_i = os.path.join(directory, "grid_structure")

    edges_route_df = pd.read_csv(os.path.join(path_i, "EDGES_ROUTE.csv"))
    edges_red_df = pd.read_csv(os.path.join(path_i, "EDGES_RED.csv"))
    edges_org_df = pd.read_csv(os.path.join(path_i, "EDGES_ORG.csv"))
    transedges_df = pd.read_csv(os.path.join(path_i, "TRANSEDGES.csv"))
    os_graph_df = pd.read_csv(os.path.join(path_i, "OSGRAPH.csv"))
    os_list_df = pd.read_csv(os.path.join(path_i, "OSLIST.csv"))
    tree_df = pd.read_csv(os.path.join(path_i, "tree.csv"))

    print(
        f" - EDGES_ORG: edge (n={ len(edges_org_df)}) with corresponding source and destination, "
        f'edges ({len(np.unique(edges_org_df["EDGES_ORG.edge"]))}), '
        f'railkeys ({len(np.unique(edges_org_df["EDGES_ORG.FROM_RAILKEY"].append(edges_org_df["EDGES_ORG.TO_RAILKEY"])))}), '
        f'open connections ({len(edges_org_df["EDGES_ORG.TO_NETOPENING"]=="open")}:{len(edges_org_df["EDGES_ORG.TO_NETOPENING"]=="dicht")}), '
        f'mean impedance, = {np.mean(edges_org_df["EDGES_ORG.IMPEDANCE"])}, '
        f'mean reactance = {np.mean(edges_org_df["EDGES_ORG.REACTANCE"])}, '
        f'mean I_nom = {np.mean(edges_org_df["EDGES_ORG.I_NOM"])}, '
    )

    print(
        f' - EDGES_ROUTE: edge ({len(edges_route_df["EDGES_ROUTE.EDGE"])}) with corresponding route connection, '
        f'unique routes ({len(np.unique(edges_route_df["EDGES_ROUTE.ROUTE"]))})'
    )  #: {np.unique(edges_route_df["EDGES_ROUTE.ROUTE"])} ')

    print(
        f" - OSLIST: os with corresponding rails in the network, "
        f' unique rails ({len(np.unique(os_list_df["OSLIST.OS_rail"]))}): {np.unique(os_list_df["OSLIST.OS_rail"])} '
    )

    print(
        f' - OSGRAPH: connection between os, from {np.unique(os_graph_df["OSGRAPH.FROM_OS"])} to {np.unique(os_graph_df["OSGRAPH.TO_OS"])}, with edge number {np.unique(os_graph_df["OSGRAPH.edge"])}'
    )

    # node features ----------
    print("---> node/vertices features ", "")
    power_cons = []
    node_degree = []
    u_now = []
    u_closed = []
    for folder in folders:
        if not "fold_" in folder:
            continue
        for iteration in os.listdir(os.path.join(directory, folder)):
            path_i = os.path.join(directory, folder, iteration)
            msr_list = pd.read_csv(os.path.join(path_i, "new_MSRLIST.csv"))

            datadir = "/".join(path_i.split("/")[:-2])
            os_list = pd.read_csv(os.path.join(datadir, "grid_structure/OSLIST.csv"))
            os_list.columns = [column.split("OSLIST.")[1] for column in os_list]

            # insert dummy values for POWER_CONSUMPTION, Minimum Voltage Capacity, Maximum Voltage Capacity
            os_list["POWER_CONSUMPTION"] = 0
            os_list["Minimum Voltage Capacity"] = 8925
            os_list["Maximum Voltage Capacity"] = 12075
            os_list["init_U_MSR"] = 10500
            os_list["closed_U_MSR"] = 10500

            node_df = pd.concat(
                [
                    msr_list.rename(columns={"MSR_rail": "RAIL"}),
                    os_list.rename(columns={"OS_rail": "RAIL"}),
                ],
                ignore_index=True,
                sort=False,
            )

            edges_org = pd.read_csv(os.path.join(path_i, "new_EDGES_ORG.csv"))

            # count outgoing edges
            merged = node_df.merge(edges_org, left_on="RAIL", right_on="FROM_RAILKEY")
            counts = (
                merged.groupby("RAIL")
                .edge.count()
                .reindex(node_df.RAIL.unique(), fill_value=0)
            )
            node_df = node_df.merge(counts, on="RAIL")
            node_df.rename(columns={"edge": "outgoing"}, inplace=True)

            # count incoming edges
            merged = node_df.merge(edges_org, left_on="RAIL", right_on="TO_RAILKEY")
            counts = (
                merged.groupby("RAIL")
                .edge.count()
                .reindex(node_df.RAIL.unique(), fill_value=0)
            )
            node_df = node_df.merge(counts, on="RAIL")
            node_df.rename(columns={"edge": "incoming"}, inplace=True)

            # add incoming and outgoing edges to get degree
            node_df["degree"] = node_df["incoming"] + node_df["outgoing"]
            node_df.drop(columns=["incoming", "outgoing"], inplace=True)

            index2railkey = node_df["RAIL"]
            railkey2index = pd.Series(index2railkey.index.values, index=index2railkey)

            node_df = node_df[
                ["POWER_CONSUMPTION", "init_U_MSR", "closed_U_MSR", "degree"]
            ]

            node_df = node_df.astype("float64")

            power_cons.append(node_df["POWER_CONSUMPTION"])
            node_degree.append(node_df["degree"])
            u_now.append(node_df["init_U_MSR"])
            u_closed.append(node_df["closed_U_MSR"])

    print("Mean power consumption: ", np.mean(np.array(power_cons)))
    power_h = np.mean(np.array(power_cons), axis=0)
    power_no0 = power_h[np.where(power_h != 0.001)[0]]
    plt.hist(power_no0, bins=30)
    plt.title(f" mean distribution power consumption of dataset {dataset_explore}")
    # plt.show()
    plt.savefig(
        f"logs/figures/mean distribution power consumption of dataset {dataset_explore}.png"
    )

    print("Mean closed U: ", np.mean(np.array(u_closed)))
    u_closed_h = np.mean(np.array(u_closed), axis=0)
    u_closed_no0 = u_closed_h[np.where(u_closed_h != 0.001)[0]]
    plt.hist(u_closed_no0, bins=30)
    plt.title(f" mean distribution closed U of dataset {dataset_explore}")
    # plt.show()
    plt.savefig(f"logs/figures/mean distribution closed U of dataset {dataset_explore}.png")

    print("Mean current U: ", np.mean(np.array(u_now)))
    u_now_h = np.mean(np.array(u_now), axis=0)
    u_now_no0 = u_now_h[np.where(u_now_h != 0.001)[0]]
    plt.hist(u_now_no0, bins=30)
    plt.title(f" mean distribution current U of dataset {dataset_explore}")
    # plt.show()
    plt.savefig(f"logs/figures/mean distribution current U of dataset {dataset_explore}.png")

    print("Mean node degree: ", np.mean(np.array(node_degree)))
    node_degree_h = np.mean(np.array(node_degree), axis=0)
    node_degree_no0 = node_degree_h[np.where(node_degree_h != 0.001)[0]]
    plt.hist(node_degree_no0, bins=30)
    plt.title(f" mean distribution node degree of dataset {dataset_explore}")
    # plt.show()
    plt.savefig(
        f"logs/figures/mean distribution node degree of dataset {dataset_explore}.png"
    )

    print(f"\n")
