import re

class Rule():
    def __init__(self, rule_id, head, body, conf, pred_k=None):
        self.id = rule_id
        self.head = head
        self.body = body
        self.conf = conf
        if pred_k is not None:
            self.pred_k = pred_k

    def __eq__(self, other):
        if isinstance(other, Rule):
            return self.id == other.id and self.head == other.head and self.body == other.body
        return False
    
    def __hash__(self):
        return hash((self.id, self.head, tuple(self.body)))
    
    def __len__(self):
        return len(self.body)
    
    def __str__(self):
        return str(self.id) + " " + str(self.head) + " :- " + str(self.body) + " conf: " + str(self.conf)


class MGNNRule():
    def __init__(self, rule_id, head, body, conf, is_inverse, arity, text=None):
        self.id = rule_id
        self.head = head
        self.body = body
        self.conf = conf
        self.text = text

        self.is_inverse = is_inverse
        self.arity = arity

    def __len__(self):
        return len(self.body)


class RuleSet():
    def __init__(self, rules):
        self.rules = rules
        self.rules_grouped_by_head = {}
        for i, rule in enumerate(rules):
            if rule.head not in self.rules_grouped_by_head:
                self.rules_grouped_by_head[rule.head] = []
            self.rules_grouped_by_head[rule.head].append(i)

    def materialize(self, facts_with_conf, sub2obj, obj2sub):
        con_pairs = {}
        for x in sub2obj:
            objs = sub2obj[x]
            subs = obj2sub[x] if x in obj2sub else set()
            con_pairs[x] = subs | objs
        for x in obj2sub:
            if x not in con_pairs:
                con_pairs[x] = obj2sub[x]

        output_facts_with_conf = {}

        for fact in facts_with_conf:
            h, r, t = fact
            conf = facts_with_conf[fact]
            if conf == 1:
                output_facts_with_conf[fact] = conf
            else:
                output_facts_with_conf[fact] = 0
                if r == 'rdf:type':
                    for rule_index in self.rules_grouped_by_head[t]:
                        rule = self.rules[rule_index]
                        if rule.id == 'R5':
                            if (h, 'rdf:type', rule.body) in facts_with_conf:
                                body_conf = facts_with_conf[(h, 'rdf:type', rule.body)]
                                output_facts_with_conf[fact] += body_conf * rule.conf
                        elif rule.id == 'R6':
                            for y in con_pairs[h]:
                                if (y, 'rdf:type', rule.body) in facts_with_conf:
                                    body_conf = facts_with_conf[(y, 'rdf:type', rule.body)]
                                    output_facts_with_conf[fact] += body_conf * rule.conf
                        elif rule.id == 'R7':
                            for obj in sub2obj[h]:
                                if (h, rule.body, obj) in facts_with_conf:
                                    body_conf = facts_with_conf[(h, rule.body, obj)]
                                    output_facts_with_conf[fact] += body_conf * rule.conf
                        elif rule.id == 'R8':
                            for sub in obj2sub[h]:
                                if (sub, rule.body, h) in facts_with_conf:
                                    body_conf = facts_with_conf[(sub, rule.body, h)]
                                    output_facts_with_conf[fact] += body_conf * rule.conf
                else:
                    for rule_index in self.rules_grouped_by_head[r]:
                        rule = self.rules[rule_index]
                        body_conf = 0
                        if rule.id == 'R1':
                            if (h, rule.body, t) in facts_with_conf:
                                body_conf = facts_with_conf[(h, rule.body, t)]
                        elif rule.id == 'R2':
                            if (t, rule.body, h) in facts_with_conf:
                                body_conf = facts_with_conf[(t, rule.body, h)]
                        elif rule.id == 'R3':
                            if (h, 'rdf:type', rule.body) in facts_with_conf:
                                body_conf = facts_with_conf[(h, 'rdf:type', rule.body)]
                        elif rule.id == 'R4':
                            if (t, 'rdf:type', rule.body) in facts_with_conf:
                                body_conf = facts_with_conf[(t, 'rdf:type', rule.body)]
                        output_facts_with_conf[fact] += body_conf * rule.conf
                output_facts_with_conf[fact] = min(1, output_facts_with_conf[fact])
                        
        return output_facts_with_conf
    
    def reasoning(self, facts_with_conf, sub2obj, obj2sub, num_steps=2):
        for _ in range(num_steps):
            facts_with_conf = self.materialize(facts_with_conf, sub2obj, obj2sub)
        return facts_with_conf


pattern = re.compile(r'<(\S*?)>\[(.*?),(.*?)\]')
pattern = re.compile(r'<(\S*?)>\[(.*?)\]')

li0 = ['?A', '?B', '?A', '?B']
li1 = ['?A', '?B', '?B', '?A']
li2 = ['?B', '?A', '?B', '?A']

li3 = ['?A', '?B', '?A', '?C']
li4 = ['?A', '?B', '?C', '?A']
li5 = ['?A', '?B', '?B', '?C']
li6 = ['?A', '?B', '?C', '?B']
li7 = ['?B', '?A', '?A', '?C']
li8 = ['?B', '?A', '?C', '?A']
li9 = ['?B', '?A', '?B', '?C']
li10 = ['?B', '?A', '?C', '?B']
li_all = [li0, li1, li2, li3, li4, li5, li6, li7, li8, li9, li10]

def rule_parser_mgnn(line, conf):
    result = pattern.findall(line)
    # print(line)
    # print(result)

    rule_id = None
    if len(result)==2:
        arity = []
        is_inverse = []

        head, head_items = result[0]
        head_items = head_items.split(',')
        body, body_items = result[1]
        body_items = body_items.split(',')
        if len(head_items)==1:
            assert len(body_items)==2
            arity = [1, 2]
            if body_items[0] == '?A':
                rule_id = 'R4'
                is_inverse = False
            else:
                rule_id = 'R5'
                is_inverse = True
        elif len(head_items)==2:
            arity = [2, 2]
            head_sub, head_obj = head_items[0], head_items[1]
            body_sub, body_obj = body_items[0], body_items[1]
            if [head_sub, head_obj] == [body_sub, body_obj]:
                rule_id = 'R1'
                is_inverse = [False]
            elif [head_sub, head_obj] == [body_obj, body_sub]:
                rule_id = 'R2'
                is_inverse = [True]
        rule = MGNNRule(rule_id, head, [body], conf, is_inverse, arity, line)
    else:
        assert len(result)==3
        
        arity = []
        is_inverse = []
        head, head_items = result[0]
        head_items = head_items.split(',')
        body1, body1_items = result[1]
        body2, body2_items = result[2]
        body1_items = body1_items.split(',')
        body2_items = body2_items.split(',')
        arity1, arity2 = len(body1_items), len(body2_items)

        if arity1 == 2 and arity2 == 1:
            body1, body2 = body2, body1
            body1_items, body2_items = body2_items, body1_items
            arity1, arity2 = len(body1_items), len(body2_items)

        if len(head_items) == 1:
            rule_body = [body1, body2]
            assert 'C' not in set(body1_items+body2_items)
            if arity1 == 1 and arity2 == 1:
                assert head_items[0] == '?A' and body1_items[0] == '?A' and body2_items[0] == '?A'
                rule_id = 'type-A-A-A'  #'R7'
                is_inverse = [False, False]
                arity = [1, 1, 1]
            elif arity1 == 1 and arity2 == 2:
                arity = [1, 1, 2]
                if body1_items[0] == '?A':
                    if body2_items[0] =='?A':
                        rule_id = 'type-A-A-AB'
                        is_inverse = [False, False]
                    else:
                        rule_id = 'type-A-A-BA'
                        is_inverse = [False, True]
                else:
                    if body2_items[0] =='?A':
                        rule_id = 'type-A-B-AB'
                        is_inverse = [True, False]
                    else:
                        rule_id = 'type-A-B-BA'
                        is_inverse = [True, True]
            else:
                arity = [1, 2, 2]
                vars_li = [body1_items[0], body1_items[1], body2_items[0], body2_items[1]]
                vars_li_inv = [body2_items[0], body2_items[1], body1_items[0], body1_items[1]]
                rule_id = None
                rule_body = [body1, body2]
                for i, li in enumerate(li_all):
                    if vars_li == li:
                        rule_id = 'type-conj' + str(i)
                        is_inverse = [body1_items[0]=='?B', body2_items[0]=='?B']
                        break
                    elif vars_li_inv == li:
                        rule_id = 'type-conj' + str(i)
                        rule_body = [body2, body1]
                        is_inverse = [body1_items[0]=='?B', body2_items[0]=='?B']
                        break
        else:
            assert len(head_items) == 2
            if arity1 == 2 and arity2 == 2:
                arity = [2, 2, 2]
            
                vars_li = [body1_items[0], body1_items[1], body2_items[0], body2_items[1]]
                vars_li_inv = [body2_items[0], body2_items[1], body1_items[0], body1_items[1]]
                rule_id = None
                rule_body = [body1, body2]
                for i, li in enumerate(li_all):
                    if vars_li == li:
                        rule_id = 'conj' + str(i)
                        is_inverse = [body1_items[0]=='?B', body2_items[0]=='?B']
                        break
                    elif vars_li_inv == li:
                        rule_id = 'conj' + str(i)
                        rule_body = [body2, body1]
                        is_inverse = [body1_items[0]=='?B', body2_items[0]=='?B']
                        break
                assert rule_id is not None
            else:
                assert arity1 == 1 and arity2 == 2
                rule_body = [body1, body2]
                arity = [2, 1, 2]
                if body1_items[0] == '?A':
                    if body2_items[0] =='?A':
                        rule_id = 'type-AB-A-AB'
                        is_inverse = [False, False]
                    else:
                        rule_id = 'type-AB-A-BA'
                        is_inverse = [False, True]
                else:
                    if body2_items[0] =='?A':
                        rule_id = 'type-AB-B-AB'
                        is_inverse = [True, False]
                    else:
                        rule_id = 'type-AB-B-BA'
                        is_inverse = [True, True]

        rule = MGNNRule(rule_id, head, rule_body, conf, is_inverse, arity, line)

    return rule


def rule_parser_indigo(line, rule_id):
    conf = 0
    head = None
    body = None
    if rule_id == 'pattern1' or rule_id == 'pattern2':
        conf, body, head = line.strip().split('\t')
        body = [body]
    elif rule_id == 'pattern3':
        conf, head = line.strip().split('\t')
        body = [head]
    elif rule_id == 'pattern4':
        conf, body1, body2, head = line.strip().split('\t')
        body = [body1, body2]
    elif rule_id == 'pattern5':
        conf, body1, body2, head = line.strip().split('\t')
        body = [body1, body2]
    elif rule_id == 'pattern6':
        conf, body1, body2, head = line.strip().split('\t')
        body = [body1, body2]
    return Rule(rule_id, head, body, float(conf))


def normalize_2(head, body1):
    if 'inv_' in head:
        head = head[4:]
        body1 = 'inv_' + body1 if 'inv_' not in body1 else body1[4:]
    return head, body1


def normalize_3(head, body1, body2):
    if 'inv_' in head:
        head = head[4:]
        body1 = 'inv_' + body1 if 'inv_' not in body1 else body1[4:]
        body2 = 'inv_' + body2 if 'inv_' not in body2 else body2[4:]
    return head, body1, body2


def rule_parser_ncrl(line, threshold):
    items = line.strip().split()
    conf = float(items[0])
    if conf < threshold:
        return None
    head = items[2]
    if not ',' in line:
        body1 = items[4]
        head, body1 = normalize_2(head, body1)
        if not 'inv_' in body1:
            rule_id = 'R1'
        else:
            rule_id = 'R2'
            body1 = body1[4:]
        rule = Rule(rule_id, head, [body1], conf)
    else:
        body1 = items[4][:-1]
        body2 = items[5]
        head, body1, body2 = normalize_3(head, body1, body2)
        is_inv = ['inv_' in body1, 'inv_' in body2]
        if is_inv == [False, False]:
            rule_id = 'cp1'
        elif is_inv == [False, True]:
            rule_id = 'cp2'
            body2 = body2[4:]
        elif is_inv == [True, False]:
            rule_id = 'cp3'
            body1 = body1[4:]
        else:
            rule_id = 'cp4'
            body1 = body1[4:]
            body2 = body2[4:]
        rule = Rule(rule_id, head, [body1, body2], conf)
    return rule

def rule_parser_drum_mm(line):
    items = line.strip().split(' :  ')
    conf = float(items[2])
    head = items[0]
    body = items[1][1:-1].split(', ')
    if len(body) == 1:
        body1 = body[0].replace("'","")
        if not 'inv_' in body1:
            rule_id = 'R1'
        else:
            rule_id = 'R2'
            body1 = body1[4:]
        rule = Rule(rule_id, head, [body1], conf)
    else:
        assert len(body) == 2
        body1 = body[0].replace("'","")
        body2 = body[1].replace("'","")
        is_inv = ['inv_' in body1, 'inv_' in body2]
        if is_inv == [False, False]:
            rule_id = 'cp1'
        elif is_inv == [False, True]:
            rule_id = 'cp2'
            body2 = body2[4:]
        elif is_inv == [True, False]:
            rule_id = 'cp3'
            body1 = body1[4:]
        else:
            rule_id = 'cp4'
            body1 = body1[4:]
            body2 = body2[4:]
        rule = Rule(rule_id, head, [body1, body2], conf)
    return rule


    