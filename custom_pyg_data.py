# create a custom pygeometric dataset

import itertools
import os
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

datasets = {
    "location1": "location1",
    "location5": "location5",
    "location2": "location2",
    "location3": "location3",
    "location4": "location4",
    "all": "all",
}

class GridDataset:
    """
    load a dataset of medium voltage grids
    """

    def __init__(
        self,
        directory,
        val_fold=0,
        samples_per_fold=2000,
        small_part=False,
        normalise=False,
        different_topo=False,
        undirected=False,
    ):
        """
        args:
            directory: directory where the dataset is saved
            val_fold: cross-validation fold for validation
            samples_per_fold: amount of samples for each fold (data saving)
            small_part (string): whether to use only a small part of the data for testing grid changes [nmin1, nonnmin1, False]
        """

        self.directory = directory
        self.samples_per_fold = samples_per_fold
        self.small_part = small_part
        self.different_topo = different_topo
        self.undirected = undirected

        self.train_graphs = []
        self.val_graphs = []
        self.test_graphs = []

        self.val_fold = val_fold
        self.load_train_example()
        self.load_val_example()
        self.load_test_example()
        if normalise:
            self.normalise_data()

    def load_train_example(self):
        """
        load training data (example)
        """

        print("loading training data (example)")
        paths = []
        for fold in os.listdir(self.directory):
            if not "fold_" in fold or int(fold.split("_")[1]) == self.val_fold:
                continue
            path = os.path.join(self.directory, fold)
            for iteration in os.listdir(path)[: self.samples_per_fold]:
                if not "Iteration_" in iteration:
                    continue
                paths.append(os.path.join(path, iteration))

        self.train_graphs = self.get_graph_format(
            paths, different_topo=self.different_topo
        )
        self.train_graphs_true = self.get_graph_format(paths, different_topo=False)

        self.train_targets = self.get_targets(paths)

    def load_val_example(self):
        """
        load training data (example)
        """

        print("loading validation data (example)")
        val_dir = os.path.join(self.directory, "fold_" + str(self.val_fold))
        paths = []

        for iteration in os.listdir(val_dir)[: self.samples_per_fold]:
            if not "Iteration_" in iteration:
                continue
            paths.append(os.path.join(val_dir, iteration))

        self.val_graphs = self.get_graph_format(
            paths, different_topo=self.different_topo
        )

        self.val_targets = self.get_targets(paths)

    def load_test_example(self):
        """
        load training data (example)
        """

        print("loading testing data (example)")
        test_dir = os.path.join(self.directory, "test_data")
        paths = []

        for iteration in os.listdir(test_dir)[: self.samples_per_fold]:
            if not "Iteration_" in iteration:
                continue
            paths.append(os.path.join(test_dir, iteration))

        self.test_graphs = self.get_graph_format(paths, different_topo=False)

        self.test_targets = self.get_targets(paths)

    def normalise_data(self):
        """
        for data_i in self.train_graphs:
            print(data_i.x)
            break
        """

        print("normalise all the data")
        all_nodes = []
        all_edges = []

        for i, graph in enumerate(self.train_graphs_true):
            all_nodes.append(graph.x.detach().numpy())
            all_edges.append(graph.edge_attr.detach().numpy())
        nodes_mean = np.mean(all_nodes, axis=0) + 1e-10
        nodes_std = np.std(all_nodes, axis=0) + 1e-10
        edges_mean = np.mean(all_edges, axis=0) + 1e-10
        edges_std = np.std(all_edges, axis=0) + 1e-10

        for graph_list in [self.train_graphs, self.val_graphs, self.test_graphs]:
            diff_total_nodes = 0
            for i, graph in enumerate(graph_list):
                diff_len_nodes = len(graph.x) - len(nodes_mean)
                nodes_mean_ = nodes_mean
                nodes_std_ = nodes_std

                if diff_len_nodes < 0:
                    # print(f"total number of nodes removed for graph {i}: {diff_len_nodes*-1}")
                    diff_total_nodes += diff_len_nodes * -1
                    for _ in range(diff_len_nodes * -1):
                        nodes_mean_ = np.delete(nodes_mean_, -1, 0)
                        nodes_std_ = np.delete(nodes_std_, -1, 0)
                else:
                    # print(f"total number of nodes added for graph {i}: {diff_len_nodes}")
                    diff_total_nodes += diff_len_nodes
                    for _ in range(diff_len_nodes):
                        nodes_mean_ = np.append(nodes_mean_, [nodes_mean_[-1]], axis=0)
                        nodes_std_ = np.append(nodes_std_, [nodes_std_[-1]], axis=0)

                graph.x = torch.tensor(
                    (graph.x.detach().numpy() - nodes_mean_) / nodes_std_,
                    dtype=torch.float,
                )

                diff_len_edges = len(graph.edge_attr) - len(edges_mean)
                edges_mean_ = edges_mean
                edges_std_ = edges_std

                if diff_len_edges < 0:
                    for _ in range(diff_len_edges * -1):
                        edges_mean_ = np.delete(edges_mean_, -1, 0)
                        edges_std_ = np.delete(edges_std_, -1, 0)
                else:
                    for _ in range(diff_len_edges):
                        edges_mean_ = np.append(edges_mean_, [edges_mean_[-1]], axis=0)
                        edges_std_ = np.append(edges_std_, [edges_std_[-1]], axis=0)

                graph.edge_attr = torch.tensor(
                    (graph.edge_attr.detach().numpy() - edges_mean_) / edges_std_,
                    dtype=torch.float,
                )
            print(
                f"average number of nodes removed/added over all graph: {diff_total_nodes/len(graph_list)}"
            )

    def get_graph_format(self, path, different_topo):
        # ToDo: dislaimer, not the updated version for data augmentation
        """
        returns graph format for pygeometric

        args:
            path: path to graphs
            different_topo: boolean value to indicate if topology should differ
        returns:
            list of graph dictionaries
        """

        graphs_list = []

        for path_i in path:
            # get a list of nodes and a dictionary from railkey to index
            y_targets = torch.tensor(self.get_targets([path_i]), dtype=torch.float)

            if (
                y_targets == 0.0 and different_topo
            ):  # removing edge/node pairs when different_topo=True
                nodes_topo, railkey2index_topo, _ = self.get_nodes(path_i, remove_topo=True)
                edges_topo, senders_topo, receivers_topo = self.get_edges(path_i, railkey2index_topo)
            elif (
                y_targets == 1.0 and different_topo
            ):  # adding edge/node pairs when different_topo=True
                nodes_topo, railkey2index_topo, edges_org_topo = self.get_nodes(path_i, add_topo=True)
                edges_topo, senders_topo, receivers_topo = self.get_edges_extra(
                    edges_org_topo, railkey2index_topo
                )
            nodes, railkey2index, _ = self.get_nodes(path_i)
            edges, senders, receivers = self.get_edges(path_i, railkey2index)

            x = torch.tensor(nodes, dtype=torch.float)
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            edge_attr = torch.tensor(edges, dtype=torch.float)

            if self.undirected:
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr, reduce="mean"
                )
            graphs = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_targets)
            graphs_list.append(graphs)

            if different_topo:
                x = torch.tensor(nodes_topo, dtype=torch.float)
                edge_index = torch.tensor([senders_topo, receivers_topo], dtype=torch.long)
                edge_attr = torch.tensor(edges_topo, dtype=torch.float)
                edge_index, edge_attr = torch_geometric.utils.to_undirected(
                    edge_index, edge_attr, reduce="mean"
                )
                graphs = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_targets)
                graphs_list.append(graphs)

        return graphs_list

    def get_nmin1_iterations(self):
        """
        returns the indices of all iterations that are n-1 in the original Aalbuhn dataset
        """

        nmin1_iterations = []
        for folder in os.listdir("example_data/aalbuhn"):
            if folder == "grid_structure":
                continue
            for iteration in os.listdir(os.path.join("example_data/location1", folder)):
                unswitchables = len(
                    pd.read_csv(
                        os.path.join(
                            "example_data/location1",
                            folder,
                            iteration,
                            "unswitchables_by_capacities.csv",
                        )
                    )
                )
                if unswitchables == 0:
                    nmin1_iterations.append(iteration)
        return nmin1_iterations

    def get_targets(self, paths):
        """
        get array containing the targets for the given paths

        args:
            list of paths to the targets
        returns:
            array of targets where
            - 0 encodes not n-1
            - 1 encodes n-1
        """

        targets = []
        for i, folder in enumerate(paths):
            # read the file that contains all edges that are unswitchable because of capacities
            if len(folder) < 2:
                unswitchables = pd.read_csv(
                    os.path.join(paths, "unswitchables_by_capacities.csv")
                )
            else:
                unswitchables = pd.read_csv(
                    os.path.join(folder, "unswitchables_by_capacities.csv")
                )

            # if there are 0 unswitchable edges, the grid is n-1
            if len(unswitchables) == 0:
                targets.append(1.0)
            else:
                targets.append(0.0)

        return np.reshape(targets, [-1, 1])

    def get_targets_edge(self, paths, input_data, edge_indexs=False):
        """
        get array containing the targets for the given paths

        args:
            list of paths to the targets
        returns:
            array of targets where
            - 0 encodes not n-1
            - 1 encodes n-1
        """

        targets = []
        for i, folder in enumerate(paths):
            # read the file that contains all edges that are unswitchable because of capacities
            if len(folder) < 2:
                unswitchables = pd.read_csv(
                    os.path.join(paths, "unswitchables_by_capacities.csv")
                )
            else:
                unswitchables = pd.read_csv(
                    os.path.join(folder, "unswitchables_by_capacities.csv")
                )

            if edge_indexs:
                edges_data_i = len(input_data)
            else:
                edges_data_i = len(input_data[i].edge_index[1])
            edges_i = np.empty(edges_data_i)
            if len(unswitchables) == 0:
                # amount of edges
                edges_i.fill(1.0)
                targets.append(edges_i)
            else:
                edges_i.fill(1.0)
                location_nmin1 = unswitchables["unswitchables_by_capacities.edge"]
                for edge_nmin1 in location_nmin1:
                    edges_i[edge_nmin1] = 0.0
                targets.append(edges_i)
        return targets

    def add_degree(self, path, node_df, max_ = 20):
        # ToDo: dislaimer, not the updated version for data augmentation
        """
        add the node degree as node feature to the node_df

        args:
            path: path of the current iteration
            node_df: pd.DataFrame for addition of node degree
        returns:
            node_df: pd.DataFrame with the added node degree
        """

        edges_org = pd.read_csv(os.path.join(path, "new_EDGES_ORG.csv"))

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
        candidates_ = node_df.loc[node_df["outgoing"] <= 1] #0
        node_df["degree"] = node_df["incoming"] + node_df["outgoing"]
        node_df.drop(columns=["incoming", "outgoing"], inplace=True)

        # remove or add edges (closed netopening)
        max_sample = len(candidates_)
        sample_random = random.randint(
            0, min(max_sample, max_)
        )
        candidate_nodes_edges = candidates_.sample(n=sample_random)

        return node_df, candidate_nodes_edges

    def recompute_feat(self, graph, graph_aug, node_features, edge_features):
        # ToDo: dislaimer, not the updated version for data augmentation
        """
        recomputes feature values (nodes and edges), based on data augmentation to the topology

        args:
            graph
            graph_aug: augmented graph structure
            node_features
            edge_features

        returns:
            node_features
            edge_features
        """
        raise NotImplementedError
        #return [node_features, edge_features]

    def get_nodes(self, path, remove_topo=False, add_topo=False):
        # ToDo: dislaimer, not the updated version for data augmentation
        """
        read and return a list of node features
        - node features: [POWER_CONSUMPTION, init_U_MSR, closed_U_MSR, degree]

        args:
            path: path to the current iteration
        returns:
            nodes: list of nodes
            railkey2index: dictionary of translations from railkeys to new node indices
        """

        # read dataframes containing msr and os
        msr_list = pd.read_csv(os.path.join(path, "new_MSRLIST.csv"))
        datadir = "/".join(path.split("/")[:-2])
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

        node_df, candidates_nodes_edges = self.add_degree(path, node_df)
        index_candidates = candidates_nodes_edges.index
        node_pairs = []

        if remove_topo:
            node_df_topo = node_df.drop(list(index_candidates), axis=0, inplace=False)
        elif add_topo:
            node_df_topo = node_df
            for cand_i in list(index_candidates):
                rail_i = list(node_df_topo.iloc[[cand_i], 0])[0]
                node_df_topo = node_df_topo.append(
                    node_df_topo.loc[[cand_i]].assign(**{"RAIL": rail_i * 10}),
                    ignore_index=True,
                )
                #node_df = recompute_feat()
                #for i in node_df_topo["incoming"].loc[[cand_i]]:
                    #node_df_topo.loc[[cand_i]]=np.mean(node_df_topo["incoming"][:])
                node_pairs.append([rail_i, rail_i * 10])

        if remove_topo or add_topo:
            node_df = node_df_topo

        index2railkey = node_df["RAIL"]
        railkey2index = pd.Series(index2railkey.index.values, index=index2railkey)
        node_df = node_df[["POWER_CONSUMPTION", "init_U_MSR", "closed_U_MSR", "degree"]]
        node_df = node_df.astype("float64")
        # create list of lists, each inner list represents one node
        nodes = node_df.values.tolist()

        edges_org = 0

        if add_topo:
            edges_org = pd.read_csv(os.path.join(path, "new_EDGES_ORG.csv"))
            edge_n = len(edges_org) + 2
            for i, cand_i in enumerate(list(index_candidates)):
                sender_i, receiver_i = node_pairs[i]
                edge_n += 1
                edges_org = edges_org.append(
                    edges_org[edges_org.TO_RAILKEY == sender_i].assign(
                        **{
                            "edge": edge_n + 1,
                            "FROM_RAILKEY": sender_i,
                            "TO_RAILKEY": receiver_i,
                        }
                    ),
                    ignore_index=True,
                )

        return nodes, railkey2index, edges_org

    def get_edges_extra(self, file, railkey2index):
        # ToDo: dislaimer, not the updated version for data augmentation
        """
        read in and return lists of edge features, senders and receivers as required to create graph_dicts
        - receivers[i] receives edges[i] from senders[i]
        - edge features: [IMPEDANCE, REACTANCE, I_NOM, TO_NETOPENING, init_I_cable, closed_I_cable,
                            init_I_cable_/I_NOM, closed_I_cable_/I_NOM]

        args:
            path: path of the current iteration
            railkey2index: dictionary of translations from railkeys to new node indices
        returns:
            edges: list of edge features.
            senders: list of node indices that are the senders.
            receivers: list of node indices that receive the edges
        """

        edges_org = file
        senders = edges_org["FROM_RAILKEY"].tolist()  # get senders from edges_org
        receivers = edges_org["TO_RAILKEY"].tolist()

        receivers_ = []
        rmv_rail_idx = []
        rmv_railkey = []
        for railkey in receivers:
            try:
                receivers_.append(railkey2index[railkey])
            except:
                rmv_railkey.append(railkey)
                index_rail = receivers.index(railkey)
                rmv_rail_idx.append(index_rail)
        receivers = receivers_

        edges_org = edges_org.loc[~edges_org["TO_RAILKEY"].isin(rmv_railkey)]

        rmv_senders_idx = []

        for index_rail in sorted(rmv_rail_idx, reverse=True):
            del senders[index_rail]

        senders_ = []
        for railkey in senders:
            try:
                senders_.append(railkey2index[railkey])
            except:
                index_rail = senders.index(railkey)
                rmv_senders_idx.append(index_rail)
        senders = senders_

        edges_org = edges_org.loc[~edges_org["edge"].isin(rmv_senders_idx)]

        edges_org["init_I_cable_/I_NOM"] = (
            edges_org["init_I_cable"] / edges_org["I_NOM"]
        )
        edges_org["closed_I_cable_/I_NOM"] = (
            edges_org["closed_I_cable"] / edges_org["I_NOM"]
        )

        # change values of TO_NETOPENING from string to int ("open"->0, "dicht"->1)
        edges_org["TO_NETOPENING"] = edges_org["TO_NETOPENING"].apply(
            lambda s: 0 if s == "open" else 1
        )

        edges_org.drop(
            ["edge", "ROUTE_NAAM", "FROM_RAILKEY", "TO_RAILKEY"],
            axis="columns",
            inplace=True,
        )

        edges_org = edges_org.astype("float64")

        edges = edges_org.values.tolist()

        return edges, senders, receivers

    def get_edges(self, path, railkey2index):
        # ToDo: dislaimer, not the updated version for data augmentation
        """
        read in and return lists of edge features, senders and receivers as required to create graph_dicts
        - receivers[i] receives edges[i] from senders[i]
        - edge features: [IMPEDANCE, REACTANCE, I_NOM, TO_NETOPENING, init_I_cable, closed_I_cable,
                            init_I_cable_/I_NOM, closed_I_cable_/I_NOM]

        args:
            path: path of the current iteration
            railkey2index: dictionary of translations from railkeys to new node indices
        returns:
            edges: list of edge features.
            senders: list of node indices that are the senders.
            receivers: list of node indices that receive the edges
        """

        edges_org = pd.read_csv(os.path.join(path, "new_EDGES_ORG.csv"))
        senders = edges_org["FROM_RAILKEY"].tolist()  # get senders from edges_org
        receivers = edges_org["TO_RAILKEY"].tolist()

        receivers_ = []
        rmv_rail_idx = []
        rmv_railkey = []
        for railkey in receivers:
            try:
                receivers_.append(railkey2index[railkey])
            except:
                rmv_railkey.append(railkey)
                index_rail = receivers.index(railkey)
                rmv_rail_idx.append(index_rail)
        receivers = receivers_

        edges_org = edges_org.loc[~edges_org["TO_RAILKEY"].isin(rmv_railkey)]

        rmv_senders_idx = []

        for index_rail in sorted(rmv_rail_idx, reverse=True):
            del senders[index_rail]

        senders_ = []
        for railkey in senders:
            try:
                senders_.append(railkey2index[railkey])
            except:
                index_rail = senders.index(railkey)
                rmv_senders_idx.append(index_rail)
        senders = senders_

        edges_org = edges_org.loc[~edges_org["edge"].isin(rmv_senders_idx)]

        edges_org["init_I_cable_/I_NOM"] = (
            edges_org["init_I_cable"] / edges_org["I_NOM"]
        )
        edges_org["closed_I_cable_/I_NOM"] = (
            edges_org["closed_I_cable"] / edges_org["I_NOM"]
        )

        # change values of TO_NETOPENING from string to int ("open"->0, "dicht"->1)
        edges_org["TO_NETOPENING"] = edges_org["TO_NETOPENING"].apply(
            lambda s: 0 if s == "open" else 1
        )

        edges_org.drop(
            ["edge", "ROUTE_NAAM", "FROM_RAILKEY", "TO_RAILKEY"],
            axis="columns",
            inplace=True,
        )

        edges_org = edges_org.astype("float64")

        edges = edges_org.values.tolist()

        return edges, senders, receivers

    def get_iterator(self, mode, batchsize):
        """
        creates an iterator over batches

        args:
            mode: ['train', 'val', 'test']
            batchsize: size of batches
        returns:
            an iterator over the batches, first retrun value is number of batches
        """

        if mode in ["train", "training"]:
            graphs = self.train_graphs
            targets = self.train_targets
        elif mode in ["val", "validation"]:
            graphs = self.val_graphs
            targets = self.val_targets
        elif mode == "test":
            graphs = self.test_graphs
            targets = self.test_targets
        elif mode == "all":
            graphs = list(
                itertools.chain(self.train_graphs, self.val_graphs, self.test_graphs)
            )
            targets = np.concatenate(
                (self.train_targets, self.val_targets, self.test_targets)
            )

        # shuffle graphs and targets
        indices = np.arange(len(targets))
        np.random.shuffle(indices)
        graphs = np.array(graphs)[indices]
        targets = np.array(targets)[indices]

        n_batches = int(np.ceil(len(targets) / batchsize))
        yield n_batches

        for batch in range(n_batches):
            # start and end indices for folders
            start = batchsize * batch
            end = batchsize * (batch + 1)

            batch_graphs = graphs[start:end]
            batch_targets = targets[start:end]

            yield batch_targets
