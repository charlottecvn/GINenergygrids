import torch
import torch.nn.functional as F
from torch.nn import Sigmoid
from torchmetrics.classification import (
    Accuracy,
    BinaryRecall,
    BinaryPrecision,
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCalibrationError,
)
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from graphnetwork.GIN_model import GIN
from sklearn.calibration import calibration_curve

def l1_l2_loss(model, l1_weight=0.001, l2_weight=0.001):
    parameters_model = []
    for param_ in model.parameters():
        parameters_model.append(param_.view(-1))
    l1_loss = l1_weight * model.compute_l1_loss(torch.cat(parameters_model))
    l2_loss = l2_weight * model.compute_l2_loss(torch.cat(parameters_model))

    return l1_loss + l2_loss

def train_model(
    model,
    loader,
    val_loader,
    test_loader,
    device="cpu",
    criterion=F.binary_cross_entropy,
    opt_lr=0.01,
    n_epochs=100,
    weightdecay=0.01,
    optimizer_="adam",
    logging="None",
    print_log=True,
    name_log="test",
    save_plots=False,
    reduction_loss="mean",
    l1_weight=0.001,
    l2_weight=0.001,
    temperature_init=1.0,
    calibrate_net=True,
    *args,
):

    print(f"---> training the GNN with edge features on {device}")

    acc_func_bin = BinaryAccuracy().to(device)

    last_loss = 100
    patience = 5
    triggertimes = 0

    logging_result = {
        "step_train_val": 0,
        "loss_train": [],
        "loss_val": [],
        "loss_test": [],
        "acc_val": [],
        "acc_test": [],
        "auc_val": [],
        "auc_test": [],
    }

    if optimizer_ == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt_lr, weight_decay=weightdecay
        )
    elif optimizer_ == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt_lr, weight_decay=weightdecay
        )
    else:
        print("this optimizer is not compatible, adam is applied")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt_lr, weight_decay=weightdecay
        )

    for epoch in range(n_epochs + 1):
        total_train_loss = 0
        total_train_acc = 0

        current_len_loader_train = 0

        # train on batches
        for i, data in enumerate(loader):
            if (
                data.edge_index[0].max() > len(data.batch) - 1
                or data.edge_index[1].max() > len(data.batch) - 1
            ):
                pass
            else:
                current_len_loader_train += 1
                data = data.to(device)
                model.train()
                optimizer.zero_grad()

                out_zero, out = model(
                    data.x, data.edge_index, data.batch, data.edge_attr
                )

                pred = out
                targets = data.y

                loss_ce = criterion(pred, targets, reduction=reduction_loss)
                l_loss = l1_l2_loss(model, l1_weight, l2_weight)
                loss = loss_ce + l_loss

                acc = acc_func_bin(pred, targets)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_acc += acc.item()

        total_val_loss, total_val_acc = test_model(
            model,
            val_loader,
            device,
            criterion,
            reduction_loss=reduction_loss,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            temperature=temperature_init,
        )

        total_val_auc = AUC_test(
            model,
            val_loader,
            device,
            criterion,
            reduction_loss=reduction_loss,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            temperature=temperature_init,
        )

        # early stopping
        current_loss = total_val_loss
        if current_loss > last_loss:
            triggertimes += 1
            if triggertimes >= patience:
                print("---> early stopping! start to test process")
                break
        else:
            triggertimes = 0
        last_loss = current_loss

        print(
            f"epoch: {epoch} | "
            f"train loss: {total_train_loss/current_len_loader_train:.2f} | "
            f"val loss: {total_val_loss:.2f} | "
            f"train acc: {total_train_acc/current_len_loader_train:.2f} | "
            f"val acc: {total_val_acc:.2f} | "
            f"val auc: {total_val_auc:.2f} "
        )

        if logging == "None":
            logging_result["step_train_val"] += 1
            logging_result["loss_train"].append(total_train_loss / len(loader))
            logging_result["loss_val"].append(total_val_loss)
            logging_result["acc_val"].append(total_val_acc)
            logging_result["auc_val"].append(total_val_auc)

    test_loss, test_acc = test_model(
        model,
        test_loader,
        device,
        criterion,
        reduction_loss=reduction_loss,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        temperature=temperature_init,
    )
    test_auc = AUC_test(
        model,
        test_loader,
        device,
        criterion,
        test=True,
        reduction_loss=reduction_loss,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        temperature=temperature_init,
    )

    if calibrate_net:
        temperature_calibrated = temperature_calibration(
            model=model, loader=test_loader, device=device, lr=1e-6
        )
        temperature_calibrated_val = temperature_calibration(
            model=model, loader=val_loader, device=device, lr=1e-6
        )
        print(f"fitted temperature scaling parameter (test): {temperature_calibrated}")
        print(
            f"fitted temperature scaling parameter (val): {temperature_calibrated_val}"
        )

        test_loss_calibrated, test_acc_calibrated = test_model(
            model,
            test_loader,
            device,
            criterion,
            reduction_loss=reduction_loss,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            temperature=temperature_calibrated,
        )
        test_auc_calibrated = AUC_test(
            model,
            test_loader,
            device,
            criterion,
            test=True,
            reduction_loss=reduction_loss,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            temperature=temperature_calibrated,
        )
    else:
        test_loss_calibrated, test_acc_calibrated = test_model(
            model,
            test_loader,
            device,
            criterion,
            reduction_loss=reduction_loss,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            temperature=temperature_init,
        )
        test_auc_calibrated = AUC_test(
            model,
            test_loader,
            device,
            criterion,
            test=True,
            reduction_loss=reduction_loss,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            temperature=temperature_init,
        )

    if print_log:
        print(
            f"test loss: {test_loss:.2f} | "
            f"test acc: {test_acc:.2f}% | "
            f"test auc: {test_auc:.2f}% | "
            f"test loss (calibrated): {test_loss_calibrated:.2f} | "
            f"test acc (calibrated): {test_acc_calibrated:.2f}% | "
            f"test auc (calibrated): {test_auc_calibrated:.2f}% | "
        )

    if logging == "None":
        logging_result["loss_test"].append(test_loss_calibrated)
        logging_result["acc_test"].append(test_acc_calibrated)
        logging_result["auc_test"].append(test_auc_calibrated)

    if print_log:
        with open(f"../logs/results/logging_result_{name_log}.pkl", "wb") as file:
            pickle.dump(logging_result, file)
            file.close()

    if save_plots:
        print("--> generating pyplots of loss and accuracy")
        print(logging_result)
        fig = px.line(
            x=range(logging_result["step_train_val"]), y=logging_result["loss_train"]
        )  # fig.add_scatter
        fig.write_html(f"../logs/figures/loss_train_{name_log}.html")

        fig = px.line(
            x=range(logging_result["step_train_val"]), y=logging_result["loss_val"]
        )
        fig.write_html(f"../logs/figures/loss_val_{name_log}.html")

        fig = px.line(
            x=range(logging_result["step_train_val"]), y=logging_result["acc_val"]
        )
        fig.write_html(f"../logs/figures/acc_val_{name_log}.html")

        fig = px.line(
            x=range(logging_result["step_train_val"]), y=logging_result["auc_val"]
        )
        fig.write_html(f"../logs/figures/auc_val_{name_log}.html")

    return model

@torch.no_grad()
def AUC_test(
    model,
    loader,
    device="cpu",
    test=False,
    temperature=None,
):
    total_auc = 0
    total_auc_2 = 0
    auc_func_bin = BinaryAUROC(thresholds=None)

    model.eval()

    current_len_loader = 0

    pred_member = []
    target_member = []

    with torch.no_grad():
        for data in loader:
            if (
                data.edge_index[0].max() > len(data.batch) - 1
                or data.edge_index[1].max() > len(data.batch) - 1
            ):
                pass
            else:
                current_len_loader += 1
                data = data.to(device)
                logits, out_sigmoid = model(
                    data.x, data.edge_index, data.batch, data.edge_attr
                )

                pred = out_sigmoid
                targets = data.y

                if temperature is not None:
                    if isinstance(temperature, float):
                        out = logits / temperature
                    else:
                        out = logits.to(device) / temperature.to(device)
                    pred = torch.sigmoid(out)

                AUC = auc_func_bin(pred, targets)

                total_auc += AUC.item()
                total_auc_2 += auc_func_bin(logits, targets)

                if test:
                    pred_member.append(pred.item())
                    target_member.append(targets.item())

        total_auc /= current_len_loader

        if test:
            total_auc = auc_func_bin(
                torch.as_tensor(pred_member), torch.as_tensor(target_member)
            )

    return total_auc


@torch.no_grad()
def test_model(
    model,
    loader,
    device="cpu",
    criterion=F.binary_cross_entropy,
    reduction_loss="mean",
    l1_weight=0.001,
    l2_weight=0.001,
    temperature=None,
):

    total_loss = 0
    total_acc = 0

    acc_func_bin = BinaryAccuracy().to(device)

    model.eval()

    current_len_loader = 0

    with torch.no_grad():
        for data in loader:
            if (
                data.edge_index[0].max() > len(data.batch) - 1
                or data.edge_index[1].max() > len(data.batch) - 1
            ):
                pass
            else:
                current_len_loader += 1
                data = data.to(device)
                logits, out_sigmoid = model(
                    data.x, data.edge_index, data.batch, data.edge_attr
                )

                pred = out_sigmoid
                targets = data.y

                if temperature is not None:
                    if isinstance(temperature, float):
                        out = logits / temperature
                    else:
                        out = logits.to(device) / temperature.to(device)
                    pred = torch.sigmoid(out)

                loss_ce = criterion(pred, targets, reduction=reduction_loss)
                l_loss = l1_l2_loss(model, l1_weight, l2_weight)
                loss = loss_ce + l_loss

                acc = acc_func_bin(pred, targets)

                total_loss += loss.item()
                total_acc += acc.item()

        total_loss /= current_len_loader
        total_acc /= current_len_loader

    return total_loss, total_acc


@torch.no_grad()
def test_model_member(
    model,
    loader,
    name_log,
    device="cpu",
    criterion=F.binary_cross_entropy,
    test=False,
    reduction_loss="mean",
    l1_weight=0.001,
    l2_weight=0.001,
    temperature=None,
):

    total_loss = 0
    total_acc = 0

    acc_func_bin = BinaryAccuracy().to(device)

    model.eval()

    current_len_loader = 0

    BS_all = 0
    pred_member = []
    target_member = []

    with torch.no_grad():
        for data in loader:
            if (
                data.edge_index[0].max() > len(data.batch) - 1
                or data.edge_index[1].max() > len(data.batch) - 1
            ):
                pass
            else:
                current_len_loader += 1
                data = data.to(device)
                logits, out_sigmoid = model(
                    data.x, data.edge_index, data.batch, data.edge_attr
                )

                pred = out_sigmoid
                targets = data.y

                if temperature is not None:
                    if isinstance(temperature, float):
                        out = logits / temperature
                    else:
                        out = logits.to(device) / temperature.to(device)
                    pred = torch.sigmoid(out)
                else:
                    out = logits

                loss_ce = criterion(pred, targets, reduction=reduction_loss)
                l_loss = l1_l2_loss(model, l1_weight, l2_weight)
                loss = loss_ce + l_loss

                BS_part = (pred - targets) ** 2

                if test:
                    pred_member.append(pred.item())
                    target_member.append(targets.item())

                acc = acc_func_bin(pred, targets)

                total_loss += loss.item()
                total_acc += acc.item()
                BS_all += BS_part

        if test:
            """
            #print(pred, out)
            #draw_reliability_graph(pred, targets, name_log=name_log)
            #print(target_member)
            ECE, MCE = ECE_MCE(torch.as_tensor(pred_member), torch.as_tensor(target_member), n_bins=10)
            print(f'ECE ({ECE}), MCE ({MCE})')
            bins_true, bins_pred = calibration_curve(target_member, pred_member, n_bins=10) #.cpu().numpy()
            #print(bins_true)
            bins_true = [0.2, 0.4, 0.6, 0.8, 1.0]
            draw_reliability_graph(bins_true, bins_pred, name_log=name_log)
            """

        print(f"current len loader (test): {current_len_loader}")

        total_loss /= current_len_loader
        total_acc /= current_len_loader
        BS_all /= current_len_loader

    return total_loss, total_acc, BS_all, pred_member

