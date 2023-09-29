# permutation feature importance

import torch
from torchmetrics import MeanSquaredError

def score_model(model, data, data_x, data_z):
    logits, out_sigmoid = model(data_x, data.edge_index, data.batch, data_z)
    pred = logits
    targets = data.y
    metric_score = MeanSquaredError()
    return metric_score(pred, targets)

def PFI_model(test_loader, model):
    score_orig = 0
    data_orig = 0
    for data in test_loader:
        if (
            data.edge_index[0].max() > len(data.batch) - 1
            or data.edge_index[1].max() > len(data.batch) - 1
        ):  # not test and
            pass
        else:
            score_ = score_model(model, data, data.x, data.edge_attr)
            score_orig += score_
            data_orig += 1
    score_orig = score_orig / data_orig
    # print(f'score original (MSE): {score_orig}')

    feat_importance = torch.empty(12, dtype=torch.float)

    node_feat_names = ['power consumption', 'init_U_MSR', 'closed_U_MSR', 'degree']
    num_node_feat = 4
    for i in range(0, num_node_feat):
        score_i = 0
        perm_d = 0
        count_pass = 0
        for d in test_loader:
            if (d.edge_index[0].max() > len(d.batch) - 1 or d.edge_index[1].max() > len(d.batch) - 1):  # not test and
                pass
                count_pass += 1
            else:
                # random permute (shuffle) column i of dataset d
                t = d.x.T[i]
                idx = torch.randperm(t.shape[0])
                t = t[idx].view(t.size())
                if torch.all(t.eq(d.x.T[i])):
                    print("true t:", torch.all(t.eq(d.x.T[i])))
                d_new_x = d.x
                d_new_x[:, i] = t
                # compute score
                score_i_d = score_model(model, d, d_new_x, d.edge_attr)
                score_i += score_i_d
                perm_d += 1

        # importance column i
        perm_imp_i = score_orig - score_i / perm_d  # len(test_loader)
        feat_importance[i] = perm_imp_i
        print(f'importance node feat {node_feat_names[i]}({i}): {perm_imp_i}')

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
                score_i_d = score_model(model, d, d.x, d_new_edge_attr)
                score_i += score_i_d
                perm_d += 1

        # importance column i
        perm_imp_i = score_orig - score_i / perm_d  # len(test_loader)
        # feat_importance[i+4] = perm_imp_i
        print(f"importance edge feat {edge_feat_names[i]}({i}): {perm_imp_i}")

    print(feat_importance)
    feat_importance = feat_importance[0:8]