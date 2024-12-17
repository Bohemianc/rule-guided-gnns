from ast import arg
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score

import os.path
import yaml
import argparse
import logging

from utils import predict_entailed_fast, load_predicates, encode_input_dataset, output_scores

# parser.add_argument('--max-iterations',
#                     nargs='?',
#                     default=None,
#                     help='Maximum number of GNN interations to test. Default is None, for infinite')


def read_triples(fn):
    triples = []
    with open(fn, 'r') as f:
        for buf in f:
            h, r, t = buf.strip().split()
            triples.append((h, r, t))
    return triples


def read_triples_with_scores(fn):
    scores = {}
    with open(fn, 'r') as f:
        for buf in f:
            h, r, t, s = buf.strip().split()
            s = float(s)
            scores[(h, r, t)] = s
    return scores


def evaluate(eval_model, test_data_output_path, scores_path, add_2hop):
    # output scores
    eval_model.eval()

    incomplete_dataset = []
    with open(test_data_path, 'r') as test_dataset:
        for line in test_dataset:
            h, r, t = line.strip().split()
            incomplete_dataset.append((h,r,t))

    examples_dataset = []
    with open(test_data_output_path, 'r') as ground_entailed:
        for line in ground_entailed:
            # if the_fact[:-1] in examples_dataset:
            #     print("Duplicate fact in ground entailed dataset: {}".format(
            #         the_fact[:-1]))
            h, r, t = line.strip().split()
            examples_dataset.append((h,r,t))
    # print(len(examples_dataset))

    scores = output_scores(eval_model,
                            binaryPredicates,
                            unaryPredicates,
                            incomplete_dataset,
                            examples_dataset,
                            device=device,
                            add_2hop=add_2hop)
    with open(scores_path, 'w') as output:
        for fact in scores:
            h,r,t = fact
            output.write(f'{h}\t{r}\t{t}\t{scores[fact]}\n')
    return scores


def calculate_metrics(y_true, y_pred, y_pred_float):
    assert(sum(y_true)>0)
    # if sum(y_pred)==0:
    #     print(y_pred_float)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred_float)

    return precision, recall, accuracy, f1, auc_pr


def get_scores_and_truths(triples, scores, truths):
    scores_bin = []
    scores_float = []
    for triple in triples:
        scores_float.append(scores[triple])
        if scores[triple] > configs["threshold"]:
            scores_bin.append(1)
        else:
            scores_bin.append(0)

    truths_list = []
    for triple in triples:
        truths_list.append(truths[triple])
    
    return np.array(scores_bin), np.array(truths_list), np.array(scores_float)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--gpu', type=int, default=0)
    argparser.add_argument('--model', type=str, default='model')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = configs["exp_name"]
    dataset = configs["dataset_name"]
    test_data_path = configs["test_graph"]
    # test_data_output_path = configs["test_examples"]
    # truths_path = configs["truths_path"]
    test_example_dir = f'data/{configs["dataset_name"]}/test'
    add_2hop = configs["add_2hop"]

    # scores_path = f'experiments/{exp_name}/scores.txt'
    model_path = f'experiments/{exp_name}/models/{args.model}.pt'

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(f'experiments/{exp_name}/runs/result.txt')
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
   
    model = torch.load(model_path).to(device)

    all_precision, all_recall, all_accuracy, all_f1, all_auc_pr = [], [], [], [], []
    for i in range(10):
        test_example_path = os.path.join(test_example_dir, f'test{i}.txt')
        scores_path = f'experiments/{exp_name}/scores-{i}.txt'
        truths_path = os.path.join(test_example_dir, f'test{i}_with_truth_values.txt')
        evaluate(model, test_example_path, scores_path, add_2hop)

        triples = read_triples(test_example_path)
        scores = read_triples_with_scores(scores_path)
        truths = read_triples_with_scores(truths_path)

        y_pred, y_true, scores_float = get_scores_and_truths(triples, scores, truths)

        # print(scores_float)

        precision, recall, accuracy, f1, auc_pr = calculate_metrics(y_true, y_pred, scores_float)
        all_precision.append(precision)
        all_recall.append(recall)
        all_accuracy.append(accuracy)
        all_f1.append(f1)
        all_auc_pr.append(auc_pr)
        # logger.info(f'precision: {precision:.3f}')
        # logger.info(f'recall: {recall:.3f}')
        # logger.info(f'accuracy: {accuracy:.3f}')
        # logger.info(f'f1_score: {f1:.3f}')
        # logger.info(f'auc_pr: {auc_pr:.3f}')

    print(f'precision: {sum(all_precision)/10.0:.3f}, recall: {sum(all_recall)/10.0:.3f}, accuracy: {sum(all_accuracy)/10.0:.3f}, f1_score: {sum(all_f1)/10.0:.3f}, auc_pr: {sum(all_auc_pr)/10.0:.3f}')
