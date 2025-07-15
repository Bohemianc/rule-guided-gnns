import pickle
import numpy as np
import torch

from rule_reasoning import Rule, RuleSet, rule_parser_mgnn, rule_parser_indigo, rule_parser_ncrl, rule_parser_drum_mm
from utils import load_predicates


def read_predicates(dataset):
    binary_preds_li, unary_preds_li = load_predicates(dataset)
    unary_dict = {}
    binary_dict = {}

    num_binary = len(binary_preds_li)
    for i, pred in enumerate(binary_preds_li):
        binary_dict[pred] = i
    for i, pred in enumerate(unary_preds_li):
        unary_dict[pred] = i + num_binary

    return binary_dict, unary_dict


def cal_sc_for_guiding(train_path, dataset, threshold_a=0.01, threshold_suppr=1):
    binary_preds, unary_preds = read_predicates(dataset)
    pred_dict = {**unary_preds, **binary_preds}
    id2pred = {pred_dict[v]: v for v in pred_dict}

    pred_to_entities = dict()
    pred_to_pairs = dict()
    one_hop_pairs = set()

    with open(train_path, 'r') as train_graph:
        for RDF_fact in train_graph:
            words = RDF_fact.strip().split()
            if words[1] == 'rdf:type':
                pid = pred_dict[words[2]]
                if pid not in pred_to_entities:
                    pred_to_entities[pid] = set()
                pred_to_entities[pid].add(words[0])
            else:
                pid = pred_dict[words[1]]
                if pid not in pred_to_pairs:
                    pred_to_pairs[pid] = set()
                pred_to_pairs[pid].add((words[0], words[2]))
                one_hop_pairs.add((words[0], words[2]))
    # print(pred_to_entities)
    # two_hop_pairs, _, _, _ = get_2_hop_pairs(one_hop_pairs)
    # two_hop_pairs = set(two_hop_pairs)
    # connected_pairs = one_hop_pairs | two_hop_pairs

    num_preds = len(pred_dict)
    num_binary = len(binary_preds)

    pred_to_pairs_inv = dict()
    for i in range(num_binary):
        pred_to_pairs_inv[i] = set()
        for (u, v) in pred_to_pairs[i]:
            pred_to_pairs_inv[i].add((v, u))

    pred_to_domain = {}
    pred_to_range = {}
    for i in range(num_binary):
        pred_to_domain[i] = set()
        pred_to_range[i] = set()
        for (u, v) in pred_to_pairs[i]:
            pred_to_domain[i].add(u)
            pred_to_range[i].add(v)

    sc_A = np.zeros((num_preds, num_preds), dtype=np.float32)
    sc_B_c1 = np.zeros((num_preds, num_preds), dtype=np.float32)
    sc_B_c2 = np.zeros((num_preds, num_preds), dtype=np.float32)
    sc_B_c3 = np.zeros((num_preds, num_preds), dtype=np.float32)
    sc_B_c4 = np.zeros((num_preds, num_preds), dtype=np.float32)
    suppr_A = np.zeros((num_preds, num_preds), dtype=np.int32)
    suppr_B_c1 = np.zeros((num_preds, num_preds), dtype=np.int32)
    suppr_B_c2 = np.zeros((num_preds, num_preds), dtype=np.int32)
    suppr_B_c3 = np.zeros((num_preds, num_preds), dtype=np.int32)
    suppr_B_c4 = np.zeros((num_preds, num_preds), dtype=np.int32)

    for i in range(num_binary):
        for j in range(num_binary):
            suppr = len(pred_to_pairs[i] & pred_to_pairs[j])
            sc_A[i][j] = suppr / len(pred_to_pairs[j])   # R1, self-loop, A left-top
            suppr_A[i][j] = suppr

            suppr = len(pred_to_pairs[i] & pred_to_pairs_inv[j])
            sc_B_c3[i][j] = suppr / len(pred_to_pairs_inv[j])  # R2, c3, B[3] left-top
            suppr_B_c3[i][j] = suppr

    for i in range(num_binary, num_preds):
        for j in range(num_binary, num_preds):
            if i not in pred_to_entities or j not in pred_to_entities:
                sc_A[i][j] = 0
            else:
                suppr = len(pred_to_entities[i] & pred_to_entities[j])
                sc_A[i][j] = suppr / len(pred_to_entities[j])    # R3, self-loop, A right-bottom
                suppr_A[i][j] = suppr

    for i in range(num_binary, num_preds):
        for j in range(num_binary):
            if i not in pred_to_entities:
                sc_B_c1[i][j] = 0
                sc_B_c2[i][j] = 0
            else:
                suppr = 0
                num_ground_body = len(pred_to_pairs[j])
                for (u, v) in pred_to_pairs[j]:
                    if u in pred_to_entities[i]:
                        suppr += 1
                sc = suppr / num_ground_body
                sc_B_c1[i][j] = sc  # R4, c1, B[1] left-bottom
                suppr_B_c1[i][j] = suppr

                suppr = 0
                for (v, u) in pred_to_pairs[j]:
                    if u in pred_to_entities[i]:
                        suppr += 1
                sc = suppr / num_ground_body
                sc_B_c2[i][j] = sc  # R5, c2, B[2] left-bottom
                suppr_B_c2[i][j] = suppr

    sc_rl1 = np.stack((sc_A, sc_B_c1, sc_B_c2, sc_B_c3, sc_B_c4))
    suppr_rl1 = np.stack((suppr_A, suppr_B_c1, suppr_B_c2, suppr_B_c3, suppr_B_c4))
    mask_rl1 = (sc_rl1 >= 0) & (suppr_rl1 >= 0)

    index_body1 = []
    index_body2 = []

    a2 = 0.5
    a2_prime = 0.8
    sc_R6, sc_R7, sc_R8, sc_R9 = {}, {}, {}, {}
    guiding_rules = {}
    for pattern in ['R6', 'R7', 'R8', 'R9']:
        guiding_rules[pattern]=[]

    # === R6 ===
    for i in range(num_binary):
        for j in range(num_binary):
            for k in range(j+1, num_binary):
                if sc_A[i,j] <= threshold_a and sc_A[i,k] <= threshold_a:
                    ground_body = pred_to_pairs[j] & pred_to_pairs[k]
                    suppr = len(pred_to_pairs[i] & ground_body)
                    sc = len(pred_to_pairs[i] & ground_body) / len(ground_body) if len(ground_body) else 0
                    if a2_prime > sc > a2 and suppr >= threshold_suppr:
                        index_body1.append((0, i, j))
                        index_body2.append((0, i, k))
                        sc_R6[(i,j,k)] = sc
                        guiding_rules['R6'].append(Rule('R6', id2pred[i], [id2pred[j], id2pred[k]], sc))

    # === R7 ===
    for i in range(num_binary, num_preds):
        for j in range(num_binary, num_preds):
            for k in range(j+1, num_preds):
                if sc_A[i,j] <= threshold_a and sc_A[i,k] <= threshold_a:
                    if i in pred_to_entities and j in pred_to_entities and k in pred_to_entities:
                        ground_body = pred_to_entities[j] & pred_to_entities[k]
                        suppr = len(pred_to_entities[i] & ground_body)
                        sc = len(pred_to_entities[i] & ground_body) / len(ground_body) if len(ground_body) else 0
                        if a2_prime > sc > a2 and suppr >= threshold_suppr:
                            index_body1.append((0, i, j))
                            index_body2.append((0, i, k))
                            sc_R7[(i,j,k)] = sc
                            guiding_rules['R7'].append(Rule('R7', id2pred[i], [id2pred[j], id2pred[k]], sc))
    
    # === R8 ===
    for i in range(num_binary, num_preds):
        for j in range(num_binary, num_preds):
            for k in range(num_binary):
                if sc_B_c4[i,j] <= threshold_a and sc_B_c1[i,k] <= threshold_a:
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if j in pred_to_entities and v in pred_to_entities[j]:
                            num_ground_body += 1
                            if i in pred_to_entities and u in pred_to_entities[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body else 0
                    if a2_prime > sc > a2 and suppr >= threshold_suppr:
                        index_body1.append((4, i, j))
                        index_body2.append((1, i, k))
                        sc_R8[(i,j,k)] = sc
                        guiding_rules['R8'].append(Rule('R8', id2pred[i], [id2pred[j], id2pred[k]], sc))

    # === R9 ===
    for i in range(num_binary, num_preds):
        for j in range(num_binary, num_preds):
            for k in range(num_binary):
                if sc_B_c4[i,j] <= threshold_a and sc_B_c2[i,k] <= threshold_a:
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if j in pred_to_entities and u in pred_to_entities[j]:
                            num_ground_body += 1
                            if i in pred_to_entities and v in pred_to_entities[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body else 0
                    if a2_prime > sc > a2 and suppr >= threshold_suppr:
                        index_body1.append((4, i, j))
                        index_body2.append((2, i, k))
                        sc_R9[(i,j,k)] = sc
                        guiding_rules['R9'].append(Rule('R9', id2pred[i], [id2pred[j], id2pred[k]], sc))
    
    sc_rl2 = []
    sc_rl2.extend(sc_R6.values())
    sc_rl2.extend(sc_R7.values())
    sc_rl2.extend(sc_R8.values())
    sc_rl2.extend(sc_R9.values())

    print(len(sc_rl2))
    print(len(sc_R6), len(sc_R7), len(sc_R8), len(sc_R9))

    # with open(f'rules_for_guiding/{dataset}.pkl', 'wb') as f:
    #     pickle.dump(guiding_rules, f)

    return sc_rl1, sc_rl2, mask_rl1, index_body1, index_body2


def get_parameters(model_path):
    model = torch.load(model_path, map_location='cpu').to('cpu')
    model.eval()
    A1 = model.conv1.A.detach().numpy()
    B1 = model.conv1.B.detach().numpy()
    A2 = model.conv2.A.detach().numpy()
    B2 = model.conv2.B.detach().numpy()

    return A1, B1, A2, B2


def get_output_rules(model_path, pred_path, threshold, threshold_rl2, threshold_prime):
    A1, B1, A2, B2 = get_parameters(model_path)

    binary_preds, unary_preds = read_predicates(pred_path)
    pred2id = {**unary_preds, **binary_preds}
    id2pred = {v: k for k, v in pred2id.items()}
    num_binary = len(binary_preds)
    num_pred = len(pred2id)

    rules_by_format = {}

    for i in range(1, 10):
        rules_by_format[f'R{i}'] = []

    # ==== R1, R3 ====
    for i in range(num_binary):
        for j in range(num_binary):
            if i!=j:
                conf = A1[i,j]
                if conf >= threshold:
                    rules_by_format['R1'].append(Rule('R1', id2pred[i], [id2pred[j]], conf))
    for i in range(num_binary, num_pred):
        for j in range(num_binary, num_pred):
            if i!=j:
                conf = A1[i,j]
                if conf >= threshold:
                    rules_by_format['R3'].append(Rule('R3', id2pred[i], [id2pred[j]], conf))

    # ==== R2 ====
    for i in range(num_binary):
        for j in range(num_binary):
            conf = B1[2,i,j]
            if conf >= threshold:
                rules_by_format['R2'].append(Rule('R2', id2pred[i], [id2pred[j]], conf))

    # ==== R4, R5 ====
    for i in range(num_binary, num_pred):
        for j in range(num_binary):
            conf = B1[0,i,j]
            if conf >= threshold:
                rules_by_format['R4'].append(Rule('R4', id2pred[i], [id2pred[j]], conf))
            conf = B1[1,i,j]
            if conf >= threshold:
                rules_by_format['R5'].append(Rule('R5', id2pred[i], [id2pred[j]], conf))

    # === R6 ===
    for i in range(num_binary):
        for j in range(num_binary):
            for k in range(j+1, num_binary):
                conf_ij = A2[i,j]
                conf_ik = A2[i,k]
                if conf_ij < threshold_prime and conf_ik < threshold_prime and conf_ij + conf_ik >= threshold_rl2:
                # if conf_ij + conf_ik >= threshold_rl2:
                    rules_by_format['R6'].append(Rule('R6', id2pred[i], [id2pred[j], id2pred[k]], conf_ij + conf_ik))
    
    # === R7 ===
    for i in range(num_binary, num_pred):
        for j in range(num_binary, num_pred):
            for k in range(j+1, num_pred):
                conf_ij = A2[i,j]
                conf_ik = A2[i,k]
                if conf_ij < threshold_prime and conf_ik < threshold_prime and conf_ij + conf_ik >= threshold_rl2:
                # if conf_ij + conf_ik >= threshold_rl2:
                    rules_by_format['R7'].append(Rule('R7', id2pred[i], [id2pred[j], id2pred[k]], conf_ij + conf_ik))

    # === R8, R9 ===
    for i in range(num_binary, num_pred):
        for j in range(num_binary, num_pred):
            for k in range(num_binary):
                conf_ij = B2[3,i,j]
                conf_ik = B2[0,i,k]
                if conf_ij < threshold_prime and conf_ik < threshold_prime and conf_ij + conf_ik >= threshold_rl2:
                # if conf_ij + conf_ik >= threshold_rl2:
                    rules_by_format['R8'].append(Rule('R8', id2pred[i], [id2pred[j], id2pred[k]], conf_ij + conf_ik))

                conf_ik = B2[1,i,k]
                if conf_ij < threshold_prime and conf_ik < threshold_prime and conf_ij + conf_ik >= threshold_rl2:
                # if conf_ij + conf_ik >= threshold_rl2:
                    rules_by_format['R9'].append(Rule('R9', id2pred[i], [id2pred[j], id2pred[k]], conf_ij + conf_ik))

    rules = []
    confidences = []

    for i in range(1,10):
        cur_rules = rules_by_format[f'R{i}']
        for rule in cur_rules:
            rules.append(rule)
            confidences.append(rule.conf)

    return rules, confidences, rules_by_format


def prepare_facts(test_graph_path, test_examples_path):
    facts_with_conf = {}
    sub2obj = {}
    obj2sub = {}

    with open(test_graph_path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split()
            facts_with_conf[(h,r,t)] = 1.0
            if r!= 'rdf:type':
                if h not in sub2obj:
                    sub2obj[h] = set()
                sub2obj[h].add(t)
                if t not in obj2sub:
                    obj2sub[t] = set()
                obj2sub[t].add(h)
    
    with open(test_examples_path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split()
            facts_with_conf[(h,r,t)] = 0.0
            if r!= 'rdf:type':
                if h not in sub2obj:
                    sub2obj[h] = set()
                sub2obj[h].add(t)
                if t not in obj2sub:
                    obj2sub[t] = set()
                obj2sub[t].add(h)
    
    return facts_with_conf, sub2obj, obj2sub
    

def output_scores(examples_path, scores_path, output):
    with open(examples_path, 'r') as f:
        examples = f.readlines()
    
    with open(scores_path, 'w') as f:
        for i, example in enumerate(examples):
            h, r, t = example.strip().split()
            if (h,r,t) in output:
                f.write(f'{example.strip()}\t{output[(h,r,t)]:.4f}\n')
            else:
                f.write(f'{example.strip()}\t0.0\n')
    
    print(f'Output scores to {scores_path}.')


def cal_sc_mgnn(train_path, pred_path, rules):
    binary_preds, unary_preds = read_predicates(pred_path)
    pred_dict = {**unary_preds, **binary_preds}
    
    pred_to_pairs = {}
    pred_to_entities = {}
    hr_to_t = {}
    tr_to_h = {}

    for i in range(len(binary_preds)):
        pred_to_pairs[i] = set()
    for i in range(len(unary_preds)):
        pred_to_entities[i+len(binary_preds)] = set()

    with open(train_path, 'r') as train_graph:
        for RDF_fact in train_graph:
            words = RDF_fact.strip().split()
            if words[1] == 'rdf:type':
                pid = pred_dict[words[2]]
                pred_to_entities[pid].add(words[0])
            else:
                pid = pred_dict[words[1]]
                pred_to_pairs[pid].add((words[0], words[2]))
                hr_to_t[(words[0], pid)] = words[2]
                tr_to_h[(words[2], pid)] = words[0]

    pred_to_pairs_inv = dict()
    for i in range(len(binary_preds)):
        pred_to_pairs_inv[i] = set()
        for (u, v) in pred_to_pairs[i]:
            pred_to_pairs_inv[i].add((v, u))

    sc_li = []
    suppr_li = []
    sc = -1
    for rule in rules:
        i = pred_dict[rule.head]
        j = pred_dict[rule.body[0]]
        if len(rule) == 1:
            if rule.id == 'R1':
                sc = len(pred_to_pairs[i] & pred_to_pairs[j]) / len(pred_to_pairs[j])
                suppr = len(pred_to_pairs[i] & pred_to_pairs[j])
            elif rule.id == 'R2':
                sc = len(pred_to_pairs[i] & pred_to_pairs_inv[j]) / len(pred_to_pairs_inv[j])
                suppr = len(pred_to_pairs[i] & pred_to_pairs_inv[j])
            elif rule.id == 'R4':
                num_ground_body = len(pred_to_pairs[j])
                suppr = 0 
                for (u, v) in pred_to_pairs[j]:
                    if u in pred_to_entities[i]:
                        suppr += 1
                sc = suppr / num_ground_body if num_ground_body else 0
            elif rule.id == 'R5':
                num_ground_body = len(pred_to_pairs[j])
                suppr = 0 
                for (u, v) in pred_to_pairs[j]:
                    if v in pred_to_entities[i]:
                        suppr += 1
                sc = suppr / num_ground_body if num_ground_body else 0    
        else:
            # rule len = 2
            k = pred_dict[rule.body[1]]
            if rule.arity == [2, 2, 2]:    # wo unary predicates
                # for rule id ['conj0', 'conj1', 'conj2']
                if rule.id in ['conj0', 'conj1', 'conj2']:
                    ground_head = pred_to_pairs[i]
                    ground_body1 = pred_to_pairs[j] if not rule.is_inverse[0] else pred_to_pairs_inv[j]
                    ground_body2 = pred_to_pairs[k] if not rule.is_inverse[1] else pred_to_pairs_inv[k]
                    ground_body = ground_body1 & ground_body2
                    sc = len(ground_head & ground_body) / len(ground_body) if len(ground_body) > 0 else 0
                    suppr = len(ground_head & ground_body)
                elif rule.id == 'conj3':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs[j]:
                        if (u, k) in hr_to_t:
                            num_C = len(hr_to_t[(u, k)])
                            num_ground_body += num_C
                            if (u, v) in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'conj4':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs[j]:
                        if (u, k) in tr_to_h:
                            num_C = len(tr_to_h[(u, k)])
                            num_ground_body += num_C
                            if (u, v) in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'conj5':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs[j]:
                        if (v, k) in hr_to_t:
                            num_C = len(hr_to_t[(v, k)])
                            num_ground_body += num_C
                            if (u, v) in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'conj6':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs[j]:
                        if (v, k) in tr_to_h:
                            num_C = len(tr_to_h[(v, k)]) 
                            num_ground_body += num_C
                            if (u, v) in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'conj7':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs_inv[j]:
                        if (u, k) in hr_to_t:
                            num_C = len(hr_to_t[(u, k)])
                            num_ground_body += num_C
                            if (u, v) in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'conj8':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs_inv[j]:
                        if (u, k) in tr_to_h:
                            num_C = len(tr_to_h[(u, k)])
                            num_ground_body += num_C
                            if (u, v) in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'conj9':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs_inv[j]:
                        if (v, k) in hr_to_t:
                            num_C = len(hr_to_t[(v, k)])
                            num_ground_body += num_C
                            if (u, v) in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'conj10':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs_inv[j]:
                        if (v, k) in tr_to_h:
                            num_C = len(tr_to_h[(v, k)])
                            num_ground_body += num_C
                            if (u, v) in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
            elif rule.arity == [2, 1, 2]:
                if rule.id == 'type-AB-A-AB':
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if u in pred_to_entities[j]:
                            num_ground_body += 1
                            if (u,v) in pred_to_pairs[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-AB-A-BA':
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if v in pred_to_entities[j]:
                            num_ground_body += 1
                            if (u,v) in pred_to_pairs_inv[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-AB-B-AB':
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if v in pred_to_entities[j]:
                            num_ground_body += 1
                            if (u,v) in pred_to_pairs[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-AB-B-BA':
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if u in pred_to_entities[j]:
                            num_ground_body += 1
                            if (u,v) in pred_to_pairs_inv[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
            elif rule.arity == [1, 1, 1]:
                assert rule.id == 'type-A-A-A'
                ground_head = pred_to_entities[i]
                ground_body1 = pred_to_entities[j]
                ground_body2 = pred_to_entities[k]
                ground_body = ground_body1 & ground_body2
                sc = len(ground_head & ground_body)/len(ground_body) if len(ground_body) else 0
                suppr = len(ground_head & ground_body)/len(ground_body)
            elif rule.arity == [1, 1, 2]:
                if rule.id == 'type-A-A-AB':
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if u in pred_to_entities[j]:
                            num_ground_body += 1
                            if u in pred_to_pairs[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-A-A-BA':
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if v in pred_to_entities[j]:
                            num_ground_body += 1
                            if v in pred_to_entities[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-B-B-AB':
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if v in pred_to_entities[j]:
                            num_ground_body += 1
                            if v in pred_to_entities[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-B-B-BA':
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if u in pred_to_entities[j]:
                            num_ground_body += 1
                            if u in pred_to_entities[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
            elif rule.arity == [1, 2, 2]:
                if rule.id in ['type-conj0', 'type-conj1', 'type-conj2']:
                    ground_body1 = pred_to_pairs[j] if not rule.is_inverse[0] else pred_to_pairs_inv[j]
                    ground_body2 = pred_to_pairs[k] if not rule.is_inverse[1] else pred_to_pairs_inv[k]
                    ground_body = ground_body1 & ground_body2
                    for (u,v) in ground_body:
                        if u in pred_to_entities[i]:
                            suppr += 1
                    sc = suppr / len(ground_body) if len(ground_body) else 0
                elif rule.id == 'type-conj3':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs[j]:
                        if (u, k) in hr_to_t:
                            num_C = len(hr_to_t[(u, k)])
                            num_ground_body += num_C
                            if u in pred_to_entities[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-conj4':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs[j]:
                        if (u, k) in tr_to_h:
                            num_C = len(tr_to_h[(u, k)])
                            num_ground_body += num_C
                            if u in pred_to_entities[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-conj5':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs[j]:
                        if (v, k) in hr_to_t:
                            num_C = len(hr_to_t[(v, k)])
                            num_ground_body += num_C
                            if u in pred_to_entities[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-conj6':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs[j]:
                        if (v, k) in tr_to_h:
                            num_C = len(tr_to_h[(v, k)]) 
                            num_ground_body += num_C
                            if u in pred_to_entities[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-conj7':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs_inv[j]:
                        if (u, k) in hr_to_t:
                            num_C = len(hr_to_t[(u, k)])
                            num_ground_body += num_C
                            if u in pred_to_pairs[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-conj8':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs_inv[j]:
                        if (u, k) in tr_to_h:
                            num_C = len(tr_to_h[(u, k)])
                            num_ground_body += num_C
                            if u in pred_to_entities[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-conj9':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs_inv[j]:
                        if (v, k) in hr_to_t:
                            num_C = len(hr_to_t[(v, k)])
                            num_ground_body += num_C
                            if u in pred_to_entities[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0
                elif rule.id == 'type-conj10':
                    num_ground_body = 0
                    suppr = 0
                    for (u, v) in pred_to_pairs_inv[j]:
                        if (v, k) in tr_to_h:
                            num_C = len(tr_to_h[(v, k)])
                            num_ground_body += num_C
                            if u in pred_to_entities[i]:
                                suppr += num_C
                    sc = suppr / num_ground_body if num_ground_body > 0 else 0

        assert sc!=-1   # rules of other patterns
        sc_li.append(sc)
        suppr_li.append(suppr)

    return sc_li, suppr_li


def cal_sc_indigo(train_path, pred_path, rules):
    binary_preds, unary_preds = read_predicates(pred_path)
    pred_dict = {**unary_preds, **binary_preds}
    
    pred_to_pairs = {}
    pred_to_entities = {}
    hr_to_t = {}

    with open(train_path, 'r') as train_graph:
        for RDF_fact in train_graph:
            words = RDF_fact.strip().split()
            if words[1] == 'rdf:type':
                pid = pred_dict[words[2]]
                if pid not in pred_to_entities:
                    pred_to_entities[pid] = set()
                pred_to_entities[pid].add(words[0])
            else:
                pid = pred_dict[words[1]]
                if pid not in pred_to_pairs:
                    pred_to_pairs[pid] = set()
                pred_to_pairs[pid].add((words[0], words[2]))
                hr_to_t[(words[0], pid)] = words[2]

    pred_to_pairs_inv = dict()
    for i in range(len(binary_preds)):
        pred_to_pairs_inv[i] = set()
        for (u, v) in pred_to_pairs[i]:
            pred_to_pairs_inv[i].add((v, u))

    sc_li = []
    suppr_li = []
    sc = 0
    for rule in rules:
        i = pred_dict[rule.head]
        j = pred_dict[rule.body[0]]
        if rule.id == 'pattern1':
            sc = len(pred_to_pairs[i] & pred_to_pairs[j]) / len(pred_to_pairs[j])
            suppr = len(pred_to_pairs[i] & pred_to_pairs[j])
        elif rule.id == 'pattern2':
            if i not in pred_to_entities or j not in pred_to_entities:
                sc = 0
                suppr = 0
            else: 
                sc = len(pred_to_entities[i] & pred_to_entities[j]) / len(pred_to_entities[j])
                suppr = len(pred_to_entities[i] & pred_to_entities[j])
        elif rule.id == 'pattern3':
            sc = len(pred_to_pairs[i] & pred_to_pairs_inv[j]) / len(pred_to_pairs_inv[j])
            suppr = len(pred_to_pairs[i] & pred_to_pairs_inv[j])
        elif rule.id == 'pattern4':
            k = pred_dict[rule.body[1]]
            num_ground_body = 0
            suppr = 0
            for (x, y) in pred_to_pairs[j]:
                if (y, k) in hr_to_t:
                    for z in hr_to_t[(y, k)]:
                        num_ground_body += 1
                        if (x, z) in pred_to_pairs[i]:
                            suppr += 1
            sc = suppr / num_ground_body if num_ground_body > 0 else 0
        elif rule.id == 'pattern5':
            k = pred_dict[rule.body[1]]
            ground_body = pred_to_pairs[j] & pred_to_pairs[k]
            sc = len(pred_to_pairs[i] & ground_body) / len(ground_body) if len(ground_body) > 0 else 0
            suppr = len(pred_to_pairs[i] & ground_body)
        elif rule.id == 'pattern6':
            k = pred_dict[rule.body[1]]
            if j not in pred_to_entities or k not in pred_to_entities:
                sc = 0
                suppr = 0
            else:
                ground_body = pred_to_entities[j] & pred_to_entities[k]
                sc = len(pred_to_entities[i] & ground_body) / len(ground_body) if len(ground_body) > 0 else 0
                suppr = len(pred_to_entities[i] & ground_body)
        sc_li.append(sc)
        suppr_li.append(suppr)

    return sc_li, suppr_li


def cal_sc_ncrl(train_path, pred_path, rules):
    binary_preds, unary_preds = read_predicates(pred_path)
    binary_preds['rdf:type'] = len(binary_preds)
    # pred_dict = {**unary_preds, **binary_preds}
    pred_dict = binary_preds
    
    pred_to_pairs = {}
    hr_to_t = {}
    tr_to_h = {}

    with open(train_path, 'r') as train_graph:
        for RDF_fact in train_graph:
            words = RDF_fact.strip().split()
            pid = pred_dict[words[1]]
            h, t = words[0], words[2]
            if pid not in pred_to_pairs:
                pred_to_pairs[pid] = set()
            pred_to_pairs[pid].add((h,t))
            if (h,pid) not in hr_to_t:
                hr_to_t[(h,pid)] = set()
            hr_to_t[(h,pid)].add(t)
            if (t,pid) not in tr_to_h:
                tr_to_h[(t,pid)] = set()
            tr_to_h[(t,pid)].add(h)

    num_preds = len(pred_dict)

    pred_to_pairs_inv = dict()
    for i in range(num_preds):
        if i in pred_to_pairs:
            pred_to_pairs_inv[i] = set()
            for (u, v) in pred_to_pairs[i]:
                pred_to_pairs_inv[i].add((v, u))

    sc_li = []
    suppr_li = []
    sc = 0
    for rule in rules:
        i = pred_dict[rule.head]
        if rule.body[0] == '':
            # print(rule.id, rule.head, rule.body, rule.conf)
            sc = 0
            suppr = 0
        else:
            j = pred_dict[rule.body[0]]
            if rule.id == 'R1':
                sc = len(pred_to_pairs[i] & pred_to_pairs[j]) / len(pred_to_pairs[j]) if len(pred_to_pairs[j]) > 0 else 0
                suppr = len(pred_to_pairs[i] & pred_to_pairs[j])
            elif rule.id == 'R2':
                sc = len(pred_to_pairs[i] & pred_to_pairs_inv[j]) / len(pred_to_pairs_inv[j]) if len(pred_to_pairs_inv[j]) > 0 else 0
                suppr = len(pred_to_pairs[i] & pred_to_pairs_inv[j]) 
            else:
                k = pred_dict[rule.body[1]]
                if rule.id == 'cp1':
                    num_ground_body = 0
                    suppr = 0
                    for (x, y) in pred_to_pairs[j]:
                        if (y, k) in hr_to_t:
                            for z in hr_to_t[(y, k)]:
                                num_ground_body += 1
                                if (x, z) in pred_to_pairs[i]:
                                    suppr += 1
                    sc = suppr / num_ground_body if num_ground_body else 0
                elif rule.id == 'cp2':
                    num_ground_body = 0
                    suppr = 0
                    for (x, y) in pred_to_pairs[j]:
                        if (y, k) in tr_to_h:
                            for z in tr_to_h[(y, k)]:
                                num_ground_body += 1
                                if (x, z) in pred_to_pairs[i]:
                                    suppr += 1
                    sc = suppr / num_ground_body if num_ground_body else 0
                elif rule.id == 'cp3':
                    num_ground_body = 0
                    suppr = 0
                    for (y, x) in pred_to_pairs[j]:
                        if (y, k) in hr_to_t:
                            for z in hr_to_t[(y, k)]:
                                num_ground_body += 1
                                if (x, z) in pred_to_pairs[i]:
                                    suppr += 1
                    sc = suppr / num_ground_body if num_ground_body else 0
                elif rule.id == 'cp4':
                    num_ground_body = 0
                    suppr = 0
                    for (y, x) in pred_to_pairs[j]:
                        if (y, k) in tr_to_h:
                            for z in tr_to_h[(y, k)]:
                                num_ground_body += 1
                                if (x, z) in pred_to_pairs[i]:
                                    suppr += 1
                    sc = suppr / num_ground_body if num_ground_body else 0
        sc_li.append(sc)
        suppr_li.append(suppr)

    return sc_li, suppr_li


def cal_sc_ours(train_path, pred_path, rules):
    binary_preds, unary_preds = read_predicates(pred_path)
    pred_dict = {**unary_preds, **binary_preds}
    
    pred_to_entities = dict()
    pred_to_pairs = dict()
    one_hop_pairs = set()

    with open(train_path, 'r') as train_graph:
        for RDF_fact in train_graph:
            words = RDF_fact.strip().split()
            if words[1] == 'rdf:type':
                pid = pred_dict[words[2]]
                if pid not in pred_to_entities:
                    pred_to_entities[pid] = set()
                pred_to_entities[pid].add(words[0])
            else:
                pid = pred_dict[words[1]]
                if pid not in pred_to_pairs:
                    pred_to_pairs[pid] = set()
                pred_to_pairs[pid].add((words[0], words[2]))
                one_hop_pairs.add((words[0], words[2]))

    num_binary = len(binary_preds)

    pred_to_pairs_inv = dict()
    for i in range(num_binary):
        pred_to_pairs_inv[i] = set()
        for (u, v) in pred_to_pairs[i]:
            pred_to_pairs_inv[i].add((v, u))

    pred_to_domain = {}
    pred_to_range = {}
    num_hr_t = {}
    num_tr_h = {}
    for i in range(num_binary):
        pred_to_domain[i] = set()
        pred_to_range[i] = set()
        for (u, v) in pred_to_pairs[i]:
            pred_to_domain[i].add(u)
            pred_to_range[i].add(v)
            if (u, i) not in num_hr_t:
                num_hr_t[(u, i)] = 0
            num_hr_t[(u, i)]  += 1
            if (v, i) not in num_tr_h:
                num_tr_h[(v, i)] = 0
            num_tr_h[(v, i)]  += 1

    sc_li = []
    suppr_li = []
    sc_R1, sc_R2, sc_R3, sc_R4, sc_R5 = [], [], [], [], []
    sc_R6, sc_R7, sc_R8, sc_R9 = [], [], [], []
    # patterns = [f'R{x}' for x in range(1, 10)]
    # for pattern in patterns:
    for rule in rules:
        sc = 0
        i = pred_dict[rule.head]
        j = pred_dict[rule.body[0]]
        if rule.id == 'R1':
            sc = len(pred_to_pairs[i] & pred_to_pairs[j]) / len(pred_to_pairs[j]) if len(pred_to_pairs[j]) else 0
            sc_R1.append(sc)
            suppr = len(pred_to_pairs[i] & pred_to_pairs[j])
        elif rule.id == 'R2':
            sc = len(pred_to_pairs[i] & pred_to_pairs_inv[j]) / len(pred_to_pairs_inv[j]) if len(pred_to_pairs_inv[j]) else 0
            sc_R2.append(sc)
            suppr = len(pred_to_pairs[i] & pred_to_pairs_inv[j])
        elif rule.id == 'R3':
            if i in pred_to_entities and j in pred_to_entities:
                sc = len(pred_to_entities[i] & pred_to_entities[j]) / len(pred_to_entities[j]) if j in pred_to_entities and len(pred_to_entities[j]) else 0
                sc_R3.append(sc)
                suppr = len(pred_to_entities[i] & pred_to_entities[j])
        elif rule.id == 'R4':
            num_ground_body = len(pred_to_pairs[j])
            suppr = 0 
            for (u, v) in pred_to_pairs[j]:
                if i in pred_to_entities and u in pred_to_entities[i]:
                    suppr += 1
            sc = suppr / num_ground_body if num_ground_body else 0
            sc_R4.append(sc)
        elif rule.id == 'R5':
            num_ground_body = len(pred_to_pairs[j])
            suppr = 0 
            for (u, v) in pred_to_pairs[j]:
                if i in pred_to_entities and v in pred_to_entities[i]:
                    suppr += 1
            sc = suppr / num_ground_body if num_ground_body else 0
            sc_R5.append(sc)
        elif rule.id == 'R6':
            k = pred_dict[rule.body[1]]
            ground_body = pred_to_pairs[j] & pred_to_pairs[k]
            sc = len(pred_to_pairs[i] & ground_body) / len(ground_body) if len(ground_body) else 0
            sc_R6.append(sc)
            suppr = len(pred_to_pairs[i] & ground_body)
        elif rule.id == 'R7':
            k = pred_dict[rule.body[1]]
            if i in pred_to_entities and j in pred_to_entities and k in pred_to_entities:
                ground_body = pred_to_entities[j] & pred_to_entities[k]
                sc = len(pred_to_entities[i] & ground_body) / len(ground_body) if len(ground_body) else 0
                suppr = len(pred_to_entities[i] & ground_body)
            else:
                sc = 0
                suppr = 0
            sc_R7.append(sc)
        elif rule.id == 'R8':
            k = pred_dict[rule.body[1]]
            num_ground_body = 0
            suppr = 0
            for (u, v) in pred_to_pairs[k]:
                if j in pred_to_entities and v in pred_to_entities[j]:
                    num_ground_body += 1
                    if i in pred_to_entities and u in pred_to_entities[i]:
                        suppr += 1
            sc = suppr / num_ground_body if num_ground_body else 0
            sc_R8.append(sc)
        elif rule.id == 'R9':
            k = pred_dict[rule.body[1]]
            num_ground_body = 0
            suppr = 0
            for (u, v) in pred_to_pairs[k]:
                if j in pred_to_entities and u in pred_to_entities[j]:
                    num_ground_body += 1
                    if i in pred_to_entities and v in pred_to_entities[i]:
                        suppr += 1
            sc = suppr / num_ground_body if num_ground_body else 0
            sc_R9.append(sc)
        sc_li.append(sc)
        suppr_li.append(suppr)

    return sc_li, suppr_li
        

def get_rule_quality(rules, scs, threshold_sc):
    pattern2id = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3, 'R5': 4, 'R6.a': 5, 'R6.b': 6, 'R7': 7, 'R8':8}
    num_q_rules = [0]*9
    num_rules = [0]*9
    for rule, sc in zip(rules, scs):
        num_rules[pattern2id[rule.id]] += 1
        if sc >= threshold_sc:
            num_q_rules[pattern2id[rule.id]] += 1
    ratio = [num_q_rules[i]/num_rules[i] if num_rules[i] else -1 for i in range(9)]

    return num_rules, num_q_rules, ratio
            

def get_quality_rules(train_path, pred_path, threshold, threshold_suppr, threshold_rl2, dataset):
    """
    R1 - R9
    """

    binary_preds, unary_preds = read_predicates(pred_path)
    pred_dict = {**unary_preds, **binary_preds}
    pid_to_predicate = {v: k for k, v in pred_dict.items()}
    num_binary = len(binary_preds)
    num_predicates = len(pred_dict)
    
    pred_to_entities = dict()
    for pid in range(num_binary, num_predicates):
        pred_to_entities[pid] = set()
    pred_to_pairs = dict()
    one_hop_pairs = set()

    with open(train_path, 'r') as train_graph:
        for RDF_fact in train_graph:
            words = RDF_fact.strip().split()
            if words[1] == 'rdf:type':
                pid = pred_dict[words[2]]
                pred_to_entities[pid].add(words[0])
            else:
                pid = pred_dict[words[1]]
                if pid not in pred_to_pairs:
                    pred_to_pairs[pid] = set()
                pred_to_pairs[pid].add((words[0], words[2]))
                one_hop_pairs.add((words[0], words[2]))

    pred_to_pairs_inv = dict()
    for i in range(num_binary):
        pred_to_pairs_inv[i] = set()
        for (u, v) in pred_to_pairs[i]:
            pred_to_pairs_inv[i].add((v, u))

    pred_to_domain = {}
    pred_to_range = {}
    num_hr_t = {}
    num_tr_h = {}
    for i in range(num_binary):
        pred_to_domain[i] = set()
        pred_to_range[i] = set()
        for (u, v) in pred_to_pairs[i]:
            pred_to_domain[i].add(u)
            pred_to_range[i].add(v)
            if (u, i) not in num_hr_t:
                num_hr_t[(u, i)] = 0
            num_hr_t[(u, i)]  += 1
            if (v, i) not in num_tr_h:
                num_tr_h[(v, i)] = 0
            num_tr_h[(v, i)]  += 1

    quality_rules = {}
    for i in range(1, 10):
        quality_rules[f'R{i}'] = []

    for i in range(num_binary):
        for j in range(num_binary):
            if i == j:
                continue
            sc = len(pred_to_pairs[i]&pred_to_pairs[j])/len(pred_to_pairs[j])
            suppr = len(pred_to_pairs[i]&pred_to_pairs[j])
            if sc > threshold and suppr >= threshold_suppr:
                rule = Rule('R1', pid_to_predicate[i], [pid_to_predicate[j]], sc)
                quality_rules['R1'].append(rule)
    
    for i in range(num_binary):
        for j in range(num_binary):
            sc = len(pred_to_pairs[i]&pred_to_pairs_inv[j])/len(pred_to_pairs_inv[j])
            suppr = len(pred_to_pairs[i]&pred_to_pairs_inv[j])
            if sc > threshold and suppr >= threshold_suppr:
                rule = Rule('R2', pid_to_predicate[i], [pid_to_predicate[j]], sc)
                quality_rules['R2'].append(rule)
    
    for i in range(num_binary, num_predicates):
        for j in range(num_binary, num_predicates):
            if i == j:
                continue
            sc = len(pred_to_entities[i]&pred_to_entities[j])/len(pred_to_entities[j]) if len(pred_to_entities[j]) else 0
            suppr = len(pred_to_entities[i]&pred_to_entities[j])
            if sc > threshold and suppr >= threshold_suppr:
                rule = Rule('R3', pid_to_predicate[i], [pid_to_predicate[j]], sc)
                quality_rules['R3'].append(rule)

    for i in range(num_binary, num_predicates):
        for j in range(num_binary):
            num_ground_body = len(pred_to_pairs[j])
            suppr = 0
            for (u, v) in pred_to_pairs[j]:
                if u in pred_to_entities[i]:
                    suppr += 1
            sc = suppr / num_ground_body if num_ground_body else 0
            if sc > threshold and suppr >= threshold_suppr:
                rule = Rule('R4', pid_to_predicate[i], [pid_to_predicate[j]], sc)
                quality_rules['R4'].append(rule)

    for i in range(num_binary, num_predicates):
        for j in range(num_binary):
            num_ground_body = len(pred_to_pairs[j])
            suppr = 0
            for (v, u) in pred_to_pairs[j]:
                if u in pred_to_entities[i]:
                    suppr += 1
            sc = suppr / num_ground_body if num_ground_body else 0
            if sc > threshold and suppr >= threshold_suppr:
                rule = Rule('R5', pid_to_predicate[i], [pid_to_predicate[j]], sc)
                quality_rules['R5'].append(rule)

    for i in range(num_binary):
        for j in range(num_binary):
            for k in range(j, num_binary):
                ground_body = pred_to_pairs[j] & pred_to_pairs[k]
                suppr = len(pred_to_pairs[i] & ground_body)
                sc = suppr / len(ground_body) if len(ground_body) else 0
                rule_ij = Rule('R1', pid_to_predicate[i], pid_to_predicate[j], 0.0)
                rule_ik = Rule('R1', pid_to_predicate[i], pid_to_predicate[j], 0.0)
                if rule_ij not in quality_rules['R1'] and rule_ik not in quality_rules['R1']:
                    if 0.8 > sc > threshold_rl2 and suppr >= threshold_suppr:
                        rule = Rule('R6', pid_to_predicate[i], [pid_to_predicate[j], pid_to_predicate[k]], sc)
                        quality_rules['R6'].append(rule)
    
    for i in range(num_binary, num_predicates):
        for j in range(num_binary, num_predicates):
            for k in range(j, num_predicates):
                ground_body = pred_to_entities[j] & pred_to_entities[k]
                suppr = len(pred_to_entities[i] & ground_body)
                sc = suppr / len(ground_body) if len(ground_body) else 0
                if 0.8 > sc > threshold_rl2 and suppr >= 50:
                    rule = Rule('R7', pid_to_predicate[i], [pid_to_predicate[j], pid_to_predicate[k]], sc)
                    quality_rules['R7'].append(rule)

    for i in range(num_binary, num_predicates):
        for j in range(num_binary, num_predicates):
            if j in pred_to_entities and i in pred_to_entities:
                for k in range(num_binary):
                    num_ground_body = 0
                    suppr = 0
                    for (u,v) in pred_to_pairs[k]:
                        if v in pred_to_entities[j]:
                            num_ground_body += 1
                            if u in pred_to_entities[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body else 0
                    if 0.8 > sc > threshold_rl2 and suppr >= 50:
                        rule = Rule('R8', pid_to_predicate[i], [pid_to_predicate[j], pid_to_predicate[k]], sc)
                        quality_rules['R8'].append(rule)

    for i in range(num_binary, num_predicates):
        for j in range(num_binary, num_predicates):
            if j in pred_to_entities and i in pred_to_entities:
                for k in range(num_binary):
                    num_ground_body = 0
                    suppr = 0
                    for (v,u) in pred_to_pairs[k]:
                        if v in pred_to_entities[j]:
                            num_ground_body += 1
                            if u in pred_to_entities[i]:
                                suppr += 1
                    sc = suppr / num_ground_body if num_ground_body else 0
                    if 0.8 > sc > threshold_rl2 and suppr >= threshold_suppr:
                        rule = Rule('R9', pid_to_predicate[i], [pid_to_predicate[j], pid_to_predicate[k]], sc)
                        quality_rules['R9'].append(rule)

    # with open(f'rules_for_guiding/GraIL-BM_{dataset}.pkl', 'rb') as f:
    #     guiding_rules_rl2 = pickle.load(f)

    # quality_rules.update(guiding_rules_rl2)

    return quality_rules         
    

def cal_percentage_and_coverage(output_rules, quality_rules):
    percentage = {}
    coverage = {}
    rule_patterns = output_rules.keys()
    nums = [[0]*3 for _ in range(len(rule_patterns))]

    R_o = []

    for i, rule_id in enumerate(rule_patterns):
        inter = set(output_rules[rule_id]) & set(quality_rules[rule_id])
        percentage[rule_id] = len(inter)/len(output_rules[rule_id]) if len(output_rules[rule_id]) else 0
        coverage[rule_id] = len(inter)/len(quality_rules[rule_id]) if len(quality_rules[rule_id]) else 0
        nums[i][0] = len(output_rules[rule_id]) - len(inter)
        nums[i][1] = len(inter)
        nums[i][2] = len(quality_rules[rule_id]) - len(inter)
        R_o.extend(set(output_rules[rule_id]) - inter)

    nums = np.array(nums)
    total = np.sum(nums, axis=0)
    tot_percentage = total[1]/(total[0] + total[1])
    tot_coverage = total[1]/(total[1] + total[2])

    return percentage, coverage, nums, tot_percentage, tot_coverage, R_o


def get_mgnn_sc(dataset, top_k=100000):
    threshold = 5
    # read MGNN rules
    if dataset == 'INDIGO-BM':
        rule_path = f'rules/mgnn/{dataset}_from-data_EC_{threshold}_extrarules.txt'   # w. types
        rule_conf_path = f'rules/mgnn/{dataset}_from-data_EC_{threshold}_extrarules_thresholds.txt'
        train_path = f'data/{dataset}/train/train.txt'
        pred_path = dataset
    else:
        rule_path = f'rules/mgnn/GraIL-BM_{dataset}_from-data_EC_{threshold}_extrarules.txt'   # w.o. types
        rule_conf_path = f'rules/mgnn/GraIL-BM_{dataset}_from-data_EC_{threshold}_extrarules_thresholds.txt'
        train_path = f'data/GraIL-BM_{dataset}/train/train.txt'
        pred_path = f'GraIL-BM_{dataset}'

    output_rules = []
    confidences = []
    with open(rule_path, 'r') as f1, open(rule_conf_path, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            conf = float(line2.strip())
            confidences.append(conf)
            rule = rule_parser_mgnn(line1.strip(), conf)
            output_rules.append(rule)
    
    if top_k < len(output_rules):
        confidences = np.array(confidences)
        indices = np.argpartition(confidences, -top_k)[-top_k:]
        filtered_output_rules = [output_rules[i] for i in indices]
    else:
        filtered_output_rules = output_rules

    sc_li, suppr_li = cal_sc_mgnn(train_path, pred_path, filtered_output_rules)

    return np.array(sc_li), np.array(suppr_li), filtered_output_rules


def get_indigo_sc(dataset, top_k=100000):
    # read INDIGO rules
    if dataset == 'INDIGO-BM':
        rule_dir = f'rules/indigo/captured_rules_0.7/{dataset}'
    else:
        rule_dir = f'rules/indigo/captured_rules_0.7/GraIL-BM_{dataset}'
        
    rules = []
    confidences = []
    with open(f'{rule_dir}/pattern1.txt', 'r') as f1:
        for line in f1:
            rule = rule_parser_indigo(line.strip(), 'pattern1')
            rules.append(rule)
            confidences.append(rule.conf)
    with open(f'{rule_dir}/pattern2.txt', 'r') as f2:
        for line in f2:
            rule = rule_parser_indigo(line.strip(), 'pattern2')
            rules.append(rule)
            confidences.append(rule.conf)
    with open(f'{rule_dir}/pattern3.txt', 'r') as f3:
        for line in f3:
            rule = rule_parser_indigo(line.strip(), 'pattern3')
            rules.append(rule)
            confidences.append(rule.conf)
    with open(f'{rule_dir}/pattern4.txt', 'r') as f4:
        for line in f4:
            rule = rule_parser_indigo(line.strip(), 'pattern4')
            rules.append(rule)
            confidences.append(rule.conf)
    with open(f'{rule_dir}/pattern5.txt', 'r') as f5:
        for line in f5:
            rule = rule_parser_indigo(line.strip(), 'pattern5')
            rules.append(rule)
            confidences.append(rule.conf)
    with open(f'{rule_dir}/pattern6.txt', 'r') as f6:
        for line in f6:
            rule = rule_parser_indigo(line.strip(), 'pattern6')
            rules.append(rule)
            confidences.append(rule.conf)

    if top_k < len(rules):
        confidences = np.array(confidences)
        indices = np.argpartition(confidences, -top_k)[-top_k:]
        filtered_rules = [rules[i] for i in indices]
    else:
        filtered_rules = rules

    sc_li, suppr_li = cal_sc_indigo(f'data/GraIL-BM_{dataset}/train/train.txt', f'GraIL-BM_{dataset}', filtered_rules)

    return np.array(sc_li), np.array(suppr_li), filtered_rules


def get_ncrl_sc(dataset, top_k=100000):
    # read NCRL rules
    threshold_ncrl = 0.01

    rules = []
    confidences = []

    rule_path1 = f'rules/ncrl/{dataset}_10_1.txt'
    rule_path2 = f'rules/ncrl/{dataset}_10_2.txt'
    with open(rule_path1) as f1:
        for line in f1:
            rule = rule_parser_ncrl(line.strip(), threshold_ncrl)
            if rule is not None:
                rules.append(rule)
                confidences.append(rule.conf)
    with open(rule_path2) as f2:
        for line in f2:
            rule = rule_parser_ncrl(line.strip(), threshold_ncrl)
            if rule is not None:
                rules.append(rule)
                confidences.append(rule.conf)

    if top_k < len(rules):
        confidences = np.array(confidences)
        indices = np.argpartition(confidences, -top_k)[-top_k:]
        filtered_rules = [rules[i] for i in indices]
    else:
        filtered_rules = rules

    sc_li, suppr_li = cal_sc_ncrl(f'data/GraIL-BM_{dataset}/train/train.txt', f'GraIL-BM_{dataset}', filtered_rules)

    return np.array(sc_li), np.array(suppr_li)


def get_drummm_sc(dataset, top_k=100000):
    # read drum-mm rules
    rules = []
    confidences = []

    rule_path = f'rules/mmdrum/{dataset}.txt'

    with open(rule_path, 'r') as f:
        for line in f:
            rule = rule_parser_drum_mm(line)
            if rule:
                rules.append(rule)
                confidences.append(rule.conf)

    if top_k < len(rules):
        confidences = np.array(confidences)
        indices = np.argpartition(confidences, -top_k)[-top_k:]
        filtered_rules = [rules[i] for i in indices]
    else:
        filtered_rules = rules

    sc_li, suppr = cal_sc_ncrl(f'data/GraIL-BM_{dataset}/train/train.txt', f'GraIL-BM_{dataset}', filtered_rules)

    return np.array(sc_li), np.array(suppr)


def get_ours_sc(dataset, top_k, threshold, threshold_rl2, threshold_prime):
    model_path = f'experiments/{dataset}_rules/models/model.pt'
    if dataset == 'INDIGO-BM':
        train_path = f'data/{dataset}/train/train_w_types.txt'
        pred_path = dataset
    else:
        train_path = f'data/GraIL-BM_{dataset}/train/train_w_types.txt'
        pred_path = f'GraIL-BM_{dataset}'
               
    rules, confidences, _ = get_output_rules(model_path, pred_path, threshold, threshold_rl2, threshold_prime)

    if top_k < len(rules):
        confidences = np.array(confidences)
        indices = np.argpartition(confidences, -top_k)[-top_k:]
        filtered_rules = [rules[i] for i in indices]
    else:
        filtered_rules = rules

    sc_li, suppr_li = cal_sc_ours(train_path, pred_path, filtered_rules)

    return np.array(sc_li), np.array(suppr_li)


if __name__=='__main__':
    datasets = ['fb237_v1',  'fb237_v2',  'fb237_v3',  'fb237_v4',  'nell_v1',  'nell_v2',  'nell_v3',  'nell_v4']

    thresholds_rl1 = [0.3 for _ in range(8)]
    thresholds_rl2 = [0.5 for _ in range(8)]
    thresholds_rl1[4] = 0.35
    thresholds_rl2[4] = 0.65
    thresholds_rl1_prime = thresholds_rl1

    THRESHOLD_SC = 0.4
    THRESHOLD_SUPPRORT = 1

    NUM_BASELINES = 1
    NUM_DATASETS = 8
    percentages_all = [[0.]*NUM_DATASETS for _ in range(NUM_BASELINES)]
    num_R_all = [[0]*NUM_DATASETS for _ in range(NUM_BASELINES)]
    num_QR_all = [[0]*NUM_DATASETS for _ in range(NUM_BASELINES)]

    top_k = 500   

# ===== compare ====

    for i in range(8):
        all_sc = []
        all_support = []    
        dataset = datasets[i]
        
        sc, suppr, _ = get_mgnn_sc(dataset, top_k)
        all_sc.append(sc)
        all_support.append(suppr)

        sc, suppr, _ = get_indigo_sc(dataset, top_k)       
        all_sc.append(sc)
        all_support.append(suppr)

        sc, suppr = get_ncrl_sc(dataset, top_k)
        all_sc.append(sc)
        all_support.append(suppr)

        sc, suppr = get_drummm_sc(dataset, top_k)
        all_sc.append(sc)
        all_support.append(suppr)

        sc, suppr = get_ours_sc(dataset, top_k, thresholds_rl1[i], thresholds_rl2[i], thresholds_rl1_prime[i])
        all_sc.append(sc)
        all_support.append(suppr)

        for j in range(NUM_BASELINES):
            num_R = len(all_sc[j])
            mask_sc = all_sc[j] >= THRESHOLD_SC
            mask_suppr = all_support[j] >= THRESHOLD_SUPPRORT
            num_QR = np.sum(mask_sc & mask_suppr)
            num_R_all[-1][i] = num_R
            num_QR_all[-1][i] = num_QR
            percentages_all[j][i] = num_QR / num_R if num_QR else 0.0
    
    print('percentage')
    for i in range(NUM_BASELINES):
        for j in range(len(percentages_all[0])):
            print(f'{percentages_all[i][j]:.4f}', end= ' ')
        print()
    print()

    print(num_R_all)
    print(num_QR_all)

# ====== Ours ======
    THRESHOLD_RL2 = 0.8
    for i in range(8):
        dataset = datasets[i]

        model_path = f'experiments/{dataset}_rules9_aaai/models/model.pt'
        if dataset == 'INDIGO-BM':
            test_graph_path = f'data/{dataset}/test/test_graph_w_types.txt'
            test_examples_path = f'data/{dataset}/test/test_all_examples.txt'
            train_path = f'data/{dataset}/train/train_w_types.txt'
            pred_path = dataset
        else:
            # test_graph_path = f'data/GraIL-BM_{dataset}/test/test_graph_w_types.txt'
            # test_examples_path = f'data/GraIL-BM_{dataset}/test/test_all_examples.txt'
            train_path = f'data/GraIL-BM_{dataset}/train/train_w_types.txt'
            pred_path = f'GraIL-BM_{dataset}'
               
        rules, confidences, output_rules_by_format = get_output_rules(model_path, pred_path, thresholds_rl1[i], thresholds_rl2[i], thresholds_rl1_prime[i])
        quality_rules = get_quality_rules(train_path, pred_path, THRESHOLD_SC, THRESHOLD_SUPPRORT, THRESHOLD_RL2, dataset)

        percentages, coverages, nums, percentage_tot, coverage_tot, R_o = cal_percentage_and_coverage(output_rules_by_format, quality_rules)

        print(f'=== {dataset} ===')
        print(percentages)
        print(coverages)
        print(nums)
        # print(nums[:,0]+nums[:,1])
        # print(nums[:,1]+nums[:,2])
        print(f'{percentage_tot:.3f}, {coverage_tot:.3f}')

        # for it in R_o:
        #     print(it)


    ## intro
    top_k = 1000000
    all_sc = [[0 for _ in range(8)] for _ in range(4)]
    avg_sc = [[0 for _ in range(8)] for _ in range(4)]
    for i in range(8):
        dataset = datasets[i]
        print(dataset)
        sc, suppr, _ = get_mgnn_sc(dataset, top_k)
        all_sc[0][i] = sc
        avg_sc[0][i] = np.mean(sc)
        sc, suppr, _ = get_indigo_sc(dataset, top_k)
        all_sc[1][i] = sc
        avg_sc[1][i] = np.mean(sc)
        sc, suppr = get_ncrl_sc(dataset, top_k)
        all_sc[2][i] = sc
        avg_sc[2][i] = np.mean(sc)
        sc, suppr = get_drummm_sc(dataset, top_k)
        all_sc[3][i] = sc
        avg_sc[3][i] = np.mean(sc)

    print(avg_sc)
    
