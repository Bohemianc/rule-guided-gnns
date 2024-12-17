import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import logging
from torch.optim import lr_scheduler

import os.path
import json
import yaml
import argparse

from utils import *
from gnn_architectures import GNN
from evaluate import calculate_metrics
from evaluate_rules import cal_sc_for_guiding


def add_constraints(model):
    for name, param in model.named_parameters():
        # if 'bias' not in name:
        param.data.clamp_(0, 1)
    with torch.no_grad():
        model.conv1.A.fill_diagonal_(1.0) 
        model.conv2.A.fill_diagonal_(1.0)


def add_mask(model):
    with torch.no_grad():
        model.conv1.A[:num_binary,num_binary:] = 0
        model.conv1.A[num_binary:,:num_binary] = 0
        model.conv1.B[0,:num_binary,:num_binary] = 0
        model.conv1.B[0,num_binary:,num_binary:] = 0
        model.conv1.B[1,:num_binary,:num_binary] = 0
        model.conv1.B[1,num_binary:,num_binary:] = 0
        model.conv1.B[2,num_binary:] = 0
        model.conv1.B[2,:num_binary,num_binary:] = 0
        model.conv1.B[3,:num_binary] = 0
        model.conv1.B[3,num_binary:,:num_binary] = 0
        model.conv1.bias_single[:num_binary] = 0
        model.conv1.bias_pair[num_binary:] = 0
        model.conv2.A[:num_binary,num_binary:] = 0
        model.conv2.A[num_binary:,:num_binary] = 0
        model.conv2.B[0,:num_binary,:num_binary] = 0
        model.conv2.B[0,num_binary:,num_binary:] = 0
        model.conv2.B[1,:num_binary,:num_binary] = 0
        model.conv2.B[1,num_binary:,num_binary:] = 0
        model.conv2.B[2,num_binary:] = 0
        model.conv2.B[2,:num_binary,num_binary:] = 0
        model.conv2.B[3,:num_binary] = 0
        model.conv2.B[3,num_binary:,:num_binary] = 0
        model.conv2.bias_single[:num_binary] = 0
        model.conv2.bias_pair[num_binary:] = 0


def rule_loss(model, sc_rl1, sc_rl2, mask_rl1, index_body1, index_body2, threshold_b):
    A1, B1 = model.conv1.A, model.conv1.B
    A2, B2 = model.conv2.A, model.conv2.B
    epsilon = 1e-6
    para1 = torch.stack((A1, B1[0], B1[1], B1[2], B1[3]))
    para2 = torch.stack((A2, B2[0], B2[1], B2[2], B2[3]))
 
    loss1 = torch.max(sc_rl1[mask_rl1] - para1[mask_rl1] + epsilon, torch.tensor(0).to(device))**2

    if len(sc_rl2) == 0:
        return loss1.mean()
    else:
        conf = para2[index_body1[:,0], index_body1[:,1], index_body1[:,2]] + para2[index_body2[:,0], index_body2[:,1], index_body2[:,2]]
        loss2 = torch.max(sc_rl2 - conf + epsilon, torch.tensor(0).to(device))**2
    
        conf_rl1 = torch.cat((para2[index_body1[:,0], index_body1[:,1], index_body1[:,2]], para2[index_body2[:,0], index_body2[:,1], index_body2[:,2]]))
        loss3 = torch.max(conf_rl1 - threshold_b - epsilon, torch.tensor(0).to(device))**2

        return loss1.mean() + loss2.mean() + loss3.mean()

def eval(mask, pred_mat, label_mat):
    indices = torch.nonzero(mask == 1)
    
    preds_float = pred_mat[indices[:,0], indices[:, 1]]
    preds = torch.where(preds_float > configs["threshold"], torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    truths = label_mat[indices[:,0], indices[:, 1]]
    preds_float = preds_float.tolist()
    preds = preds.tolist()
    truths = truths.tolist()

    precision, recall, accuracy, f1, auc_pr = calculate_metrics(truths, preds, preds_float)

    return precision, recall, accuracy, f1, auc_pr


def train(train_loader, train_mask):
    model.train()
    
    batch = train_loader.dataset[0]
    label = batch.y
    optimizer.zero_grad()

    # print(batch)
    output = model(batch)

    assert (not (output != output).any())

    train_pred = torch.mul(output, train_mask.to(device))
    loss1 = bce_loss(train_pred, label)

    if configs["add_rules"]:
        loss2 = rule_loss(model, sc_rl1, sc_rl2, mask_rl1, index_body1, index_body2, threshold_b)
        loss = loss1 + configs["rule_weight"]*loss2
    else:
        loss = loss1

    assert (not (loss != loss).any())

    loss.backward()
    optimizer.step()
    scheduler.step()

    if configs["add_constraints"]:
        add_constraints(model)

    model.eval()
    valid_output = model(batch)
    valid_pred = torch.mul(valid_output.detach(), valid_mask.to(device))
    precision_val, recall_val, acc_val, f1_val, auc_val = eval(valid_mask, valid_pred, valid_labels)

    return loss, precision_val, recall_val, acc_val, f1_val, auc_val
     

if __name__ == "__main__":
    setup_seed(42)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--gpu', type=int, default=0)

    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = configs['exp_name']
    train_path = configs["train_path"]
    valid_path = configs["valid_path"]

    bce_weight = configs["bce_weight"] if "bce_weight" in configs else 5.0
    aggr = configs["aggr"] if "aggr" in configs else "add"
    sc_threshold = configs["sc_threshold"] if "sc_threshold" in configs else 0.5
    threshold_a = configs["threshold_a"] if "threshold_a" in configs else 0.0
    threshold_b = configs["threshold_b"] if "threshold_b" in configs else 0.0

    if not os.path.exists(f'experiments/{exp_name}'):
        os.makedirs(f'experiments/{exp_name}')
        os.makedirs(f'experiments/{exp_name}/runs')
        os.makedirs(f'experiments/{exp_name}/models')
        os.makedirs(f'experiments/{exp_name}/scripts')

    clear_directory(f'experiments/{exp_name}/runs')
    save_important_files(args.config, exp_name)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(f'experiments/{exp_name}/runs/log.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    binaryPredicates, unaryPredicates = load_predicates(configs["dataset_name"])
    num_binary = len(binaryPredicates)
    num_unary = len(unaryPredicates)
    mask_threshold = num_binary + num_unary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)
    # device = torch.device('cpu')

    train_dataset = []
    with open(train_path, 'r') as f:
        for line in f:
            h,r,t,label = line.strip().split('\t')
            train_dataset.append((h,r,t,label))

    valid_dataset = []
    with open(valid_path, 'r') as f:
        for line in f:
            h,r,t,label = line.strip().split('\t')
            valid_dataset.append((h,r,t,label))

    if configs["add_rules"]:
        sc_rl1, sc_rl2, mask_rl1, index_body1, index_body2 = cal_sc_for_guiding(f'data/{configs["dataset_name"]}/train/train_w_types.txt', configs["dataset_name"], threshold_a)
        sc_rl1 = torch.tensor(sc_rl1).to(device)
        sc_rl2 = torch.tensor(sc_rl2).to(device)
        mask_rl1 = torch.tensor(mask_rl1).to(device)
        index_body1 = torch.tensor(index_body1).to(device)
        index_body2 = torch.tensor(index_body2).to(device)

    train_input_dataset, train_query_dataset = split_known(train_dataset)
    all_known_dataset = train_input_dataset
    all_unknown_dataset = valid_dataset + train_query_dataset

    (train_x, train_edge_index, train_edge_types,
    train_node_to_const_dict, train_const_to_node_dict,
    train_pred_dict, num_singleton_nodes) = encode_input_dataset(
                                                                    all_known_dataset,
                                                                    all_unknown_dataset,
                                                                    binaryPredicates,
                                                                    unaryPredicates,
                                                                    add_2hop=configs["add_2hop"],
                                                                    )
    print(train_x.sum())
    train_y, train_mask = generate_labels_and_mask(train_dataset, train_node_to_const_dict, train_const_to_node_dict, train_pred_dict)
    valid_labels, valid_mask = generate_labels_and_mask(valid_dataset, train_node_to_const_dict, train_const_to_node_dict, train_pred_dict)
    print(train_y.sum(), train_mask.sum())
    print(valid_labels.sum(), valid_mask.sum())

    train_data = Data(
        x=train_x, y=train_y, edge_index=train_edge_index, edge_type=train_edge_types)
    train_loader = DataLoader(
        dataset=[train_data.to(device)], batch_size=1)

    model = GNN(num_unary, num_binary, num_edge_types=4, num_singleton_nodes=num_singleton_nodes, num_layers=configs["num_layers"], dropout=configs["dropout"], aggr=aggr).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["learning_rate"], weight_decay=configs["weight_decay"])
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    if configs["add_constraints"]:
        add_constraints(model)

    if configs["add_mask"]:
        add_mask(model)

    min_loss = None
    num_bad_iterations = 0
    divisor = 100
    max_num_bad = 100
    best_epoch = 0
    max_f1 = 0

    print(sum(p.numel() for p in model.parameters()))
    weight = torch.where(train_y == 1, torch.tensor(bce_weight), torch.tensor(1.0)).to(device)
    bce_loss = torch.nn.BCELoss(reduction='mean', weight=weight)

    logger.info("Training...")
    for epoch in range(configs["num_epochs"]):
        loss, precision, recall, acc, f1, auc = train(train_loader, train_mask)

        if min_loss is None:
            min_loss = loss
        if f1 > max_f1:
            max_f1 = f1
            best_epoch = epoch
            torch.save(model, f'experiments/{exp_name}/models/best_model.pt')
        if epoch % divisor == 0:
            logger.info('Epoch: {:03d}, Loss: {:.5f}, Min_loss: {:.5f}, v_p: {:.4f}, v_r: {:.4f}, v_a: {:.4f}, v_f: {:.4f}, v_auc: {:.4f}'.
                format(epoch, loss, min_loss, precision, recall, acc, f1, auc))
            print('Epoch: {:03d}, Loss: {:.5f}, Min_loss: {:.5f}, v_p: {:.4f}, v_r: {:.4f}, v_a: {:.4f}, v_f: {:.4f}, v_auc: {:.4f}'.
                format(epoch, loss, min_loss, precision, recall, acc, f1, auc))

        if (epoch+1) % 100 == 0:
            torch.save(model, f'experiments/{exp_name}/models/e{epoch+1}.pt')
        if loss >= min_loss:
            num_bad_iterations += 1
            if num_bad_iterations > max_num_bad:
                print(f"Stopping early in epoch {epoch}")
                break
        else:
            num_bad_iterations = 0
            min_loss = loss

    logger.info('Best epoch: {:d}, Max_f1: {:.4f}'.format(best_epoch, max_f1))
    print('Best epoch: {:d}, Max_f1: {:.4f}'.format(best_epoch, max_f1))

    torch.save(model, f'experiments/{exp_name}/models/model.pt')
    
