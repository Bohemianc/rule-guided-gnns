# Rule-guided GNNs for Explainable KG Reasoning

This is the official code of the paper *Rule-guided GNNs for Explainable Knowledge Graph Reasoning* (AAAI'25).

## Overview
We proposed a new family of GNNs whose training can be guided by rules. This is achieved through a connection between the parameters of our GNN with various forms of rules. Specifically, our approach allows
- effective rule injection to guide the GNN training,
- efficient rule extraction by directly "**reading off**" the GNN parameters,
- a comparison between the two sets of rules to analyse the effectiveness of our rule guidance, and
- the extracted rules that are both sound and plausible.

Here are examples to demonstrate the connection between the rules and the GNN parameters.
![alt text](images/overview.png)
Consider a rule $p_i(x,y) \leftarrow p_j(x,y)$ with confidence $\alpha$, and a pair of entities $a,b$. If the fact $p_j(a,b)$ holds, then $\mathbf{h}_{a,b}^{(0)}[j]=1$ according to our KG encoding. From our rule encoding, $\mathbf{A}^{(1)}[i][j]=\alpha$. From the message passing equation, $\mathbf{h}_{a,b}^{(1)}[i] \ge \big(\mathbf{A}^{(1)}\mathbf{h}^{(0)}_{a,b}\big)[i] \ge \alpha$, which means the plausibility of the fact $p_i(a,b)$ is greater than that of $p_j(a,b)$.


## Installation

- python: 3.7.7
- pytorch: 1.12.1
- torch_geometric: 2.3.1

## Directory Structure

The following basic directory structure is required to run our implementation:

```bash
.
├── configs
├── data
    └── GraIL-BM_fb237_v1
        └── train-labeled-typed.txt
        └── valid-labeled-typed.txt
        └── test
            └── test_graph_w_types.txt
            └── test0.txt
            └── test0_with_truth_values.txt
            └── ...
            └── test9.txt
            └── test9_with_truth_values.txt
    └── ...
├── experiments
├── predicates
├── rules
```


## Reproduction
We provided the configuration files in the `configs` folder to reproduce the results in the paper.

### Training 

To train our model, run the script ```./train.py```.  The ```--config``` argument is the file path of the configurations. The `--gpu` argument indicates which gpu to use, which is optional. As an example, here is training command for the benchmark `GraIL-BM_fb237_v1`.
```bash
python train.py --config GraIL-BM_fb237_v1-train.yaml 
```
This will store the learned model in ```experiments/${experiment_name}/models/model.pt``` and save a snapshot model in `experiments/${experiment_name}/models/e{epoch}.pt` once every 100 epochs. 

### Applying a model 

To apply a model to a dataset, run the script ```./evaluate.py```. Similarly, the ```--config``` argument is the file path of the configurations for test, and the `--gpu` argument indicates which gpu to use. 

As an example, here is the command for applying the model trained for the benchmark `GraIL-BM_fb237_v1`.

```bash
python evaluate.py --config GraIL-BM_fb237_v1-eval.yaml 
```

### Rule extraction

To extract the rules captured by our model and compare the rules with those from the baselines, run the script ```./extract_rules.py```. 
```bash
 python evaluate_rules.py
```
This will output the number of extracted rules, the percentage of high-quality rules of each method for each benchmark.

## Citation

If you find the code useful in your research, please cite the following paper.

```bibtex
@inproceedings{wang2025rule,
  title={Rule-guided GNNs for Explainable Knowledge Graph Reasoning},
  author={Wang, Zhe and Ma, Suxue and Wang, Kewen and Zhuang Zhiqiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
