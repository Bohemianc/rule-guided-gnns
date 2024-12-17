import csv
import random
import time
import json
from itertools import combinations
from matplotlib.table import table
import os
import shutil
from shutil import copyfile

import torch
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as ssp
from tqdm import tqdm


RDF_type_string = 'rdf:type'


def get_2_hop_pairs(dataset, is_test=False):
    edges = []
    ent2id = dict()
    rel2id = dict()
    num_ent = 0
    num_rel = 0
    entities = []
    relations = []

    for data in dataset:
        if is_test:
            h, r, t = data
            label = '1'
        else:
            h, r, t, label = data
        if label == '0':
            continue
        if r == RDF_type_string:
            continue
        for ent in [h, t]:
            if ent not in ent2id:
                ent2id[ent] = num_ent
                num_ent += 1
                entities.append(ent)
        if r not in rel2id:
            rel2id[r] = num_rel
            num_rel += 1
            relations.append(r)
        edges.append((ent2id[h], rel2id[r], ent2id[t]))

    row = []
    col = []
    for edge in edges:
        row.append(edge[0])
        col.append(edge[2])
    row = np.array(row)
    col = np.array(col)
    adj_matrix = ssp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_ent, num_ent))
    adj_2_hop = dict((adj_matrix @ adj_matrix).todok())

    new_pairs = []
    for index in adj_2_hop:
        assert adj_2_hop[index] != 0
        if index[0]!=index[1]:
            new_pairs.append((entities[index[0]], entities[index[1]]))

    return new_pairs, adj_matrix, ent2id, entities


def load_predicates(dataset_name):
    '''Load the predicates from their file into memory, return them.'''
    # Lists to store binary and unary predicates
    binaryPredicates = []
    unaryPredicates = []

    file_path = './predicates/{}_predicates_w_types.csv'.format(dataset_name)   # with types

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Every line is of form "predicate,arity"
                pair = line.split(',')
                if int(pair[1][:-1]) == 1:  # [:-1] to get rid of \n
                    unaryPredicates.append(pair[0])
                else:
                    binaryPredicates.append(pair[0])
        return binaryPredicates, unaryPredicates

    except FileNotFoundError:
        raise FileNotFoundError(
            'Predicates csv file for {} dataset not found.'.format(dataset_name))


def encode_input_dataset(input_dataset, query_dataset, binaryPredicates, unaryPredicates,  add_2hop=True, valid_examples=None, is_test=False):
    start_time = time.time()

    num_binary = len(binaryPredicates)
    num_unary = len(unaryPredicates)
    feature_dimension = num_binary + num_unary

    pred_dict = {}
    for i, pred in enumerate(binaryPredicates):
        pred_dict[pred] = i
    for i, pred in enumerate(unaryPredicates):
        pred_dict[pred] = num_binary + i
    # print("Done in {} s.".format(time.time()-start_time))
    start_time = time.time()

    all_constants, all_pairs_of_constants = process(
        input_dataset, query_dataset, add_2hop, valid_examples, is_test)
    # print("Done in {} s.".format(time.time()-start_time))
    start_time = time.time()

    # print("Creating nodes...")
    singleton_nodes = list(all_constants)
    num_singleton_nodes = len(singleton_nodes)
    pair_nodes = set()
    for pair in all_pairs_of_constants:
        pair_nodes.add(tuple((pair)))
        pair_nodes.add((pair[1], pair[0]))
    pair_nodes = list(pair_nodes)

    const_node_dict = {const: i for i, const in enumerate(singleton_nodes)}
    nodes = singleton_nodes + pair_nodes
    # print("Done in {} s.".format(time.time() - start_time))
    start_time = time.time()

    # print("Creating list of edges...")
    edge_list = []
    edge_type_list = []
    pairs_as_nodes = set()
    # sub_dict={}
    # obj_dict={}
    for i, pair in enumerate(pair_nodes):
        #     h_id=const_node_dict[pair[0]]
        #     t_id=const_node_dict[pair[1]]
        #     if h_id not in sub_dict:
        #         sub_dict[h_id]=set()
        #     sub_dict[h_id].add(i+num_singleton_nodes)
        #     if t_id not in obj_dict:
        #         obj_dict[t_id]=set()
        #     obj_dict[t_id].add(i+num_singleton_nodes)

        # Link each pair to just the node corresponding to its first constant
        edge_list.append((const_node_dict[pair[0]], i + num_singleton_nodes))
        edge_type_list.append(0)
        edge_list.append((i + num_singleton_nodes, const_node_dict[pair[0]]))
        edge_type_list.append(0)
        edge_list.append(
            (const_node_dict[pair[1]], i + num_singleton_nodes))
        edge_type_list.append(1)
        edge_list.append(
            (i + num_singleton_nodes, const_node_dict[pair[1]]))
        edge_type_list.append(1)
        if (pair[1], pair[0]) in const_node_dict:
            # Link to reversed version of this node, of type 2
            edge_list.append(
                (const_node_dict[(pair[1], pair[0])], i + num_singleton_nodes))
            edge_type_list.append(2)
            edge_list.append(
                (i + num_singleton_nodes, const_node_dict[(pair[1], pair[0])]))
            edge_type_list.append(2)
        #  This adds mappings from pairs of constants to the corresponding pair nodes, and completes the
        #  definition of const_node_dict.
        const_node_dict[tuple((pair))] = i + num_singleton_nodes
        #  Fill the field pairs_as_nodes as expected.
        pairs_as_nodes.add(
            (const_node_dict[pair[0]], const_node_dict[pair[1]]))
        pairs_as_nodes.add(
            (const_node_dict[pair[1]], const_node_dict[pair[0]]))
    # Also link every pair of single constants for which there exists a
    # binary predicate in the dataset:
    edge_list = edge_list + list(pairs_as_nodes)
    edge_type_list = edge_type_list + [3 for _ in pairs_as_nodes]

    # for pair in inter_node_dict:
    #     k_name = inter_node_dict[pair]
    #     edge_list.append((const_node_dict[(pair[0], k_name)], const_node_dict[tuple((pair))]))
    #     edge_type_list.append(4)
    #     edge_list.append((const_node_dict[(k_name, pair[1])], const_node_dict[tuple((pair))]))
    #     edge_type_list.append(5)
    # for sub in sub_dict:
    #     node_ids=list(sub_dict[sub])
    #     for i in node_ids:
    #         for j in node_ids:
    #             if i!=j:
    #                 edge_list.append(
    #                     (i, j))
    #                 edge_type_list.append(4)
    #                 edge_list.append(
    #                     (j, i))
    #                 edge_type_list.append(4)
    # for obj in obj_dict:
    #     node_ids=list(obj_dict[obj])
    #     for i in node_ids:
    #         for j in node_ids:
    #             if i!=j:
    #                 edge_list.append(
    #                     (i, j))
    #                 edge_type_list.append(5)
    #                 edge_list.append(
    #                     (j, i))
    #                 edge_type_list.append(5)

    # print("Done in {} s.".format(time.time() - start_time))
    start_time = time.time()

    # print("Constructing additional return objects...")
    #  Return variables
    # print("Node to constant dictionary")
    node_to_const_dict = {index: constant for index,
                                              constant in enumerate(nodes)}
    # print("Edge list")
    return_edge_list = torch.LongTensor(edge_list).t().contiguous()
    # print("Edge type list")
    return_edge_type_list = torch.LongTensor(edge_type_list)
    if len(return_edge_list) == 0:
        return_edge_list = torch.LongTensor([[], []])
    # print("Graph input")
    # Now create x vectors:
    x = np.zeros((len(nodes), feature_dimension))
    for item in input_dataset:
        if is_test:
            h, r, t = item
            label = '1'
        else:
            h, r, t, label = item
        if r == RDF_type_string:
            const_index = const_node_dict[h]
            pred_index = pred_dict[t]
        else:
            const_index = const_node_dict[(h,t)]
            pred_index = pred_dict[r]
        if label == '1':
            x[const_index][pred_index] = 1
    x = torch.FloatTensor(x)
    # print("Query mask")
    # Now create x vectors:
    # print("Done in {} s.".format(time.time() - start_time))
    return x, return_edge_list, return_edge_type_list, node_to_const_dict, const_node_dict, pred_dict, num_singleton_nodes


def process(input_dataset, query_dataset, add_2hop, valid_examples=None, is_test=False):
    all_constants = set()
    all_pairs_of_constants = set()

    for RDF_triple in input_dataset+query_dataset:
        if is_test:
            h, r, t = RDF_triple
        else:
            h, r, t, label = RDF_triple
        if r == 'rdf:type':
            pred = t
            constants = h
            all_constants.add(h)
        else:
            pred = r
            constants = (h,t)
            all_pairs_of_constants.add(constants)
            all_constants.add(h)
            all_constants.add(t)

        # if label == 1:
        #     if constants not in input_dataset_constants_to_predicates_dict:
        #         input_dataset_constants_to_predicates_dict[constants] = set()
        #     input_dataset_constants_to_predicates_dict[constants].add(pred)

    if add_2hop:
        new_pairs, adj_matrix, ent2id, entities = get_2_hop_pairs(input_dataset, is_test)
        inter_node_dict = dict()
        for pair in new_pairs:
            # i, j = ent2id[pair[0]], ent2id[pair[1]]
            # s1 = set(np.argwhere(adj_matrix.getrow(i).toarray() == 1)[:,1].tolist())
            # s2 = set(np.argwhere(adj_matrix.getcol(j).toarray() == 1)[:,0].tolist())
        #     assert len(s1&s2)>0
        #     if len(s1 & s2) > 0:
        #         # k = random.choice(list(s1 & s2))
        #         #             # inter_node_dict[pair] = entities[k]
        #         # if not ((i,k) in dict(adj_matrix.todok()) and (k,j) in dict(adj_matrix.todok())):
        #         #     print('====================')
            all_pairs_of_constants.add((pair[0], pair[1]))

    if valid_examples != None:
        for RDF_triple in valid_examples:
            RDF_list = str.split(RDF_triple)
            if RDF_list[1] == 'rdf:type':
                pred = RDF_list[2]
                constants = RDF_list[0]
                # Toggled: all_constants.add(RDF_list[0])
            else:
                pred = RDF_list[1]
                constants = (RDF_list[0], RDF_list[2])
                all_pairs_of_constants.add(constants)
                all_constants.add(RDF_list[0])
                all_constants.add(RDF_list[2])

    #    if training:  # To reduce false negatives, we add in dummy constants
    #        special_constants = ['#', '##']
    #        for special_constant in special_constants:
    #            query_dataset_constants_to_predicates_dict[special_constant] = set()
    #            for constant in all_constants:
    #                all_pairs_of_constants[(special_constant, constant)] = set()
    #            all_constants.add(special_constant)

    return all_constants, all_pairs_of_constants


def generate_labels_and_mask(dataset, node_to_const_dict, const_to_node_dict,pred_dict):
    num_nodes = len(node_to_const_dict)
    num_preds = len(pred_dict)

    labels = np.zeros((num_nodes, num_preds))
    mask = np.zeros((num_nodes, num_preds))

    for item in dataset:
        h, r, t, label = item
        if r == RDF_type_string:
            const_index = const_to_node_dict[h]
            pred_index = pred_dict[t]
        else:
            const_index = const_to_node_dict[(h,t)]
            pred_index = pred_dict[r]
        mask[const_index][pred_index] = 1
        labels[const_index][pred_index] = int(label)
    return torch.FloatTensor(labels), torch.FloatTensor(mask)


def decode(node_dict, num_binary, num_unary, binaryPredicates, unaryPredicates,
           feature_vectors, threshold):
    '''Decode feature vectors back into a dataset.'''
    threshold_indices = torch.nonzero(feature_vectors >= threshold)
    GNN_dataset = set()
    for i, index in enumerate(threshold_indices):
        index = index.tolist()
        const_index = index[0]
        pred_index = index[1]
        const = node_dict[const_index]
        if type(const) is tuple:  # Then we just want to consider this if it's in the binary preds
            if pred_index < num_binary:
                predicate = binaryPredicates[pred_index]
                RDF_triplet = "{}\t{}\t{}".format(
                    const[0], predicate, const[1])
                GNN_dataset.add(RDF_triplet)
        # Then we're dealing with a unary predicate (second section of the vec)
        else:
            if pred_index >= num_binary:
                predicate = unaryPredicates[pred_index - num_binary]
                RDF_triplet = "{}\trdf:type\t{}".format(
                    const, predicate)
                GNN_dataset.add(RDF_triplet)
    return GNN_dataset


def decode_with_scores(examples, output, const_to_node_dict, pred_dict):
    scores_dict = {}
    for triple in examples:
        h, r, t = triple
        if r == RDF_type_string:
            const_index = const_to_node_dict[h]
            pred_index = pred_dict[t]
        else:
            const_index = const_to_node_dict[(h,t)]
            pred_index = pred_dict[r]
        score = output[const_index][pred_index]
        scores_dict[(h,r,t)] = score
    return scores_dict


def decode_and_get_threshold(node_dict, num_binary, num_unary, binaryPredicates, unaryPredicates,
                             feature_vectors, threshold):
    '''Decode feature vectors back into a dataset.
    Additionally report back the threshold at which all facts in the dataset would no longer be predicted'''
    threshold_indices = torch.nonzero(feature_vectors >= threshold)
    GNN_dataset = set()
    for i, index in enumerate(threshold_indices):
        index = index.tolist()
        const_index = index[0]
        pred_index = index[1]
        extraction_threshold = feature_vectors[index[0], index[1]]
        const = node_dict[const_index]
        if type(const) is tuple:  # Then we just want to consider this if it's in the binary preds
            if pred_index < num_binary:
                predicate = binaryPredicates[pred_index]
                RDF_triplet = "{} {} {}".format(const[0], predicate, const[1])
                GNN_dataset.add((RDF_triplet, extraction_threshold))
        # Then we're dealing with a unary predicate (second section of the vec)
        else:
            if pred_index >= num_binary:
                predicate = unaryPredicates[pred_index - num_binary]
                RDF_triplet = "{} rdf:type {}".format(
                    const, predicate)
                GNN_dataset.add((RDF_triplet, extraction_threshold))
    return GNN_dataset


def predict_entailed_fast(model, binaryPredicates,
                          unaryPredicates, dataset, query_dataset, max_iterations=1,
                          threshold=0.5, device='cpu'):
    '''Predict what facts are entailed by a given GNN. Use
    max_iterations = None if you want to continue until fixpoint.'''
    num_binary = len(binaryPredicates)
    num_unary = len(unaryPredicates)

    all_entailed_facts = set()
    all_facts_returned = False
    num_iterations = 1
    while not all_facts_returned:
        print("GNN iteration {}".format(num_iterations), end='\r')
        (dataset_x, edge_list, edge_type,
         node_to_const_dict, dataset_const_to_node_dict, pred_dict) = encode_input_dataset(dataset, query_dataset,
                                                                                           binaryPredicates,
                                                                                           unaryPredicates)
        test_data = Data(x=dataset_x, edge_index=edge_list,
                         edge_type=edge_type).to(device)
        entailed_facts_encoded = model(test_data)
        entailed_facts_decoded = decode(node_to_const_dict, num_binary,
                                        num_unary, binaryPredicates,
                                        unaryPredicates,
                                        entailed_facts_encoded, threshold)
        if len(entailed_facts_decoded.difference(all_entailed_facts)) == 0:
            # Then no new facts have been entailed
            all_facts_returned = True
            print('\n')
            print("No change in entailed dataset")
        else:
            all_entailed_facts = all_entailed_facts.union(
                entailed_facts_decoded)
            dataset = dataset.union(entailed_facts_decoded)
            if max_iterations is not None:
                if num_iterations >= max_iterations:
                    all_facts_returned = True
        num_iterations += 1
    return all_entailed_facts


def output_scores(model, binaryPredicates, unaryPredicates, incomplete_graph, examples, device='cpu', add_2hop=True):
    '''Give the scores for the facts in the query dataset.'''
    num_binary = len(binaryPredicates)
    num_unary = len(unaryPredicates)
    # print("Encoding input dataset...")
    (dataset_x, edge_list, edge_type,
     node_to_const_dict, const_to_node_dict, pred_dict, _) = encode_input_dataset(incomplete_graph, examples, binaryPredicates,   unaryPredicates, add_2hop=add_2hop, is_test=True)
    # print("Encapsulating input data...")
    test_data = Data(x=dataset_x, edge_index=edge_list,
                     edge_type=edge_type).to(device)
    # print("Applying model to data...")
    model.eval()
    pred = model(test_data)
    # print("Decoding...")
    scores = decode_with_scores(examples, pred, const_to_node_dict, pred_dict)
    # print("Done.")
    return scores


def load_weights(conf_paths, relation_path, type_path, dataset):
    with open(conf_paths[0], 'r') as f1, open(conf_paths[1], 'r') as f2, open(conf_paths[2], 'r') as f3, open(conf_paths[3], 'r') as f4, open(conf_paths[4]) as f5:
        A = json.loads(f1.read())
        B_c1 = json.loads(f2.read())
        B_c2 = json.loads(f3.read())
        B_c3 = json.loads(f4.read())
        B_c4 = json.loads(f5.read())

    pred2id_tot = {}
    with open(relation_path) as f:
        for line in f:
            line = line.strip().split('\t')
            pred2id_tot[line[1]] = int(line[0])
    with open(type_path) as f:
        for line in f:
            line = line.strip().split('\t')
            pred2id_tot[line[1]] = int(line[0])

    binary_preds, unary_preds = load_predicates(dataset)
    preds_sub = binary_preds + unary_preds
    num_preds_sub = len(binary_preds) + len(unary_preds)

    sc_A = torch.zeros(num_preds_sub, num_preds_sub)
    sc_B = torch.zeros(4, num_preds_sub, num_preds_sub)
    for i in range(num_preds_sub):
        for j in range(num_preds_sub):
                new_i = pred2id_tot[preds_sub[i]]
                new_j = pred2id_tot[preds_sub[j]]
                sc_A[i, j] = A[new_i][new_j]
                sc_B[0, i, j] = B_c1[new_i][new_j]
                sc_B[1, i, j] = B_c2[new_i][new_j]
                sc_B[2, i, j] = B_c3[new_i][new_j]
                sc_B[3, i, j] = B_c4[new_i][new_j]

    return sc_A, sc_B


def split_known(triples):
    
    """
    Further split the triples into 2 sets:
    1. an incomplete graph: known
    2. a set of missing facts we want to recover: unknown
    """
    # unary_triples = []
    # bin_triples = []
    # for triple in triples:
    #     if triple[1] == RDF_type_string:
    #         unary_triples.append(triple)
    #     else:
    #         bin_triples.append(triple)
    # DATA_LENGTH = len(bin_triples)
    DATA_LENGTH = len(triples)
    split_ratio = [0.9, 0.1]
    candidate = np.array(range(DATA_LENGTH))
    np.random.shuffle(candidate)
    idx_known = candidate[:int(DATA_LENGTH * split_ratio[0])]
    idx_unknown = candidate[int(DATA_LENGTH * split_ratio[0]):]
    known = []
    unknown = []
    for i in idx_known:
        known.append(triples[i])
    # known = known + unary_triples
    for i in idx_unknown:
        unknown.append(triples[i])
    return known, unknown


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def save_important_files(config_file, exp_name):
    copyfile(config_file, f'experiments/{exp_name}/configs.yaml')
    copyfile('utils.py', f'experiments/{exp_name}/scripts/utils.py')
    copyfile('train.py', f'experiments/{exp_name}/scripts/train.py')
    copyfile('gnn_architectures.py', f'experiments/{exp_name}/scripts/gnn_architectures.py')
    copyfile('evaluate.py', f'experiments/{exp_name}/scripts/evaluate.py')
    copyfile('evaluate_rules.py', f'experiments/{exp_name}/scripts/evaluate_rules.py')