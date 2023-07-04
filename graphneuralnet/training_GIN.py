# Building a graph neural network

import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, BinarySpecificity, BinaryRecall, BinaryPrecision, BinaryAccuracy
import numpy as np
import pickle
import plotly.express as px

def l1_l2_loss (model, l1_weight = 0.001, l2_weight = 0.001): #0.01
    parameters_model = []
    for param_ in model.parameters():
        parameters_model.append(param_.view(-1))
    l1_loss = l1_weight * model.compute_l1_loss(torch.cat(parameters_model))
    l2_loss = l2_weight * model.compute_l2_loss(torch.cat(parameters_model))
    
    return l1_loss+l2_loss

def train_model(
    model,
    loader,
    val_loader,
    test_loader,
    device='cpu',
    criterion=F.binary_cross_entropy,
    opt_lr=0.01,
    n_epochs=100,
    weightdecay=0.01,
    optimizer_="adam",
    logging="None",
    print_log=True,
    name_log="test",
    save_plots=True,
    reduction_loss="mean",
    l1_weight=0.001,
    l2_weight=0.001,
    edge_classification=False,
    *args,
):

    
    print(f"---> training the GNN with edge features on {device}, using edge classification ({edge_classification})")
    
    acc_func_bin = BinaryAccuracy().to(device)

    # early stopping
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
        total_val_loss = 0
        total_train_acc = 0
        total_val_acc = 0

        current_len_loader_train = 0

        # train on batches
        for i, data in enumerate(loader):
            if data.edge_index[0].max() > len(data.batch)-1 or data.edge_index[1].max() > len(data.batch)-1:
                pass
            else:
                #print(data)
                current_len_loader_train += 1
                data = data.to(device)
                model.train()
                optimizer.zero_grad()

                out_zero, out = model(
                    data.x, data.edge_index, data.batch, data.edge_attr
                )

                pred = out
                targets = data.y
                
                #print(pred.size(), targets.size())

                loss_ce = criterion(pred, targets, reduction=reduction_loss)
                l_loss = l1_l2_loss(model, l1_weight, l2_weight)
                loss = loss_ce + l_loss

                acc = acc_func_bin(pred, targets)

                loss.backward()
                optimizer.step()

                #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # gradient value clipping

                total_train_loss += loss.item()
                total_train_acc += acc.item()

        total_val_loss, total_val_acc = test_model(
            model, val_loader, device, criterion, reduction_loss=reduction_loss, l1_weight=l1_weight, l2_weight=l2_weight
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
            f"val acc: {total_val_acc:.2f} "
        )
        #print(f"--- Brier Score (mean, val): {BS_val}")

        if logging == "wandb":
            wandb.log({"train loss": total_train_loss/len(loader)})
            wandb.log({"val loss": total_val_loss})
            wandb.log({"val acc": total_val_acc})
        elif logging == "None":
            logging_result["step_train_val"] += 1
            logging_result["loss_train"].append(total_train_loss/len(loader))
            logging_result["loss_val"].append(total_val_loss)
            logging_result["acc_val"].append(total_val_acc)

    test_loss, test_acc = test_model(
        model, test_loader, device, criterion, reduction_loss=reduction_loss, l1_weight=l1_weight, l2_weight=l2_weight
    )

    if print_log:
        print(
            f"test loss: {test_loss:.2f} | "
            f"test acc: {test_acc:.2f}% | "
        )

    if logging == "wandb":
        wandb.log({"test loss": test_loss})
        wandb.log({"test acc": test_acc})
    elif logging == "None":
        logging_result["loss_test"].append(test_loss)
        logging_result["acc_test"].append(test_acc)

    if print_log:
        with open(f"../logs/results/logging_result_{name_log}.pkl", "wb") as file:
            pickle.dump(logging_result, file)
            file.close()

    if save_plots:
        print("--> generating pyplots of loss and accuracy")
        print(logging_result)
        fig = px.line(
            x=range(logging_result["step_train_val"]), y=logging_result["loss_train"]
        )
        fig.write_html(f"../logs/figures/loss_train_{name_log}.html")

        fig = px.line(
            x=range(logging_result["step_train_val"]), y=logging_result["loss_val"]
        )
        fig.write_html(f"../logs/figures/loss_val_{name_log}.html")

        fig = px.line(
            x=range(logging_result["step_train_val"]), y=logging_result["acc_val"]
        )
        fig.write_html(f"../logs/figures/acc_val_{name_log}.html")

    return model


@torch.no_grad()
def test_model(model, loader, device='cpu', criterion=F.binary_cross_entropy, reduction_loss="mean", l1_weight = 0.001, l2_weight = 0.001):
    total_loss = 0
    total_acc = 0
    
    acc_func_bin = BinaryAccuracy().to(device)
    
    model.eval()

    current_len_loader = 0
    
    with torch.no_grad():
        for data in loader:
            if data.edge_index[0].max() > len(data.batch)-1 or data.edge_index[1].max() > len(data.batch)-1:
                pass
            else:
                #print(data)
                current_len_loader += 1
                data = data.to(device)
                out_zero, out = model(data.x, data.edge_index, data.batch, data.edge_attr)

                pred = out
                targets = data.y

                loss_ce = criterion(pred, targets, reduction=reduction_loss)
                l_loss = l1_l2_loss(model, l1_weight, l2_weight)
                loss = loss_ce + l_loss

                acc = acc_func_bin(pred, targets)

                total_loss += loss.item()
                total_acc += acc.item()
            
        total_loss /= current_len_loader#len(loader)
        total_acc /= current_len_loader#len(loader)
    

    return total_loss, total_acc
    
@torch.no_grad()
def test_model_member(model, loader, device='cpu', criterion=F.binary_cross_entropy, reduction_loss="mean", l1_weight = 0.001, l2_weight = 0.001):
    total_loss = 0
    total_acc = 0
    
    acc_func_bin = BinaryAccuracy().to(device)
    
    model.eval()

    current_len_loader = 0
    
    BS_all = 0
    pred_member = []
    
    with torch.no_grad():
        for data in loader:
            if data.edge_index[0].max() > len(data.batch)-1 or data.edge_index[1].max() > len(data.batch)-1:
                pass
            else:
                #print(data)
                current_len_loader += 1
                data = data.to(device)
                out_zero, out = model(data.x, data.edge_index, data.batch, data.edge_attr)

                pred = out#_zero
                pred_sigmoid = out
                targets = data.y
                
                BS_part = (pred-targets)**2
                pred_member.append(pred)

                loss_ce = criterion(pred_sigmoid, targets, reduction=reduction_loss)
                l_loss = l1_l2_loss(model, l1_weight, l2_weight)
                loss = loss_ce + l_loss

                acc = acc_func_bin(pred_sigmoid, targets)

                total_loss += loss.item()
                total_acc += acc.item()
                BS_all += BS_part
            
        total_loss /= current_len_loader#len(loader)
        total_acc /= current_len_loader#len(loader)
        BS_all /= current_len_loader
        
    return total_loss, total_acc, BS_all, pred_member

