![zenodo 10625046](https://github.com/jackmedda/FA4GCF/assets/26059819/c37df9bc-f54b-496a-b179-b63a5de48381)# Fair Augmentation for Graph Collaborative Filtering (FA4GCF)

FA4GCF is a framework that extends the codebase of [GNNUERS](https://github.com/jackmedda/RS-BGExplainer/tree/main/gnnuers),
an approach that leverages edge-level perturbations to provide explanations of consumer unfairness and mitigate
the latter as well.
FA4GCF operates on GNN-based recommender systems as GNNUERS, but the base approach was extensively modularized,
re-adapted, and reworked to offer a simple interface for including new tools.
We focus on the adoption of GNNUERS for consumer unfairness mitigation, which leverages a set of policies that
sample the user or item set to restrict the perturbation process on specific and valuable portions of the graph.
In such a scenario, the graph perturbations are conceived only as edge additions, i.e. user-item interactions.
The edges are added only to the disadvantaged group (enjoying a lower recommendation utility), such that the
resulting overall utility of the group matches that of the advantaged group.

We provide an improved modularization for datasets (perturbed and non), models based on torch-geometric, and
sampling policies.
Compared with GNNUERS, FA4GCF better incorporates [Recbole](https://github.com/RUCAIBox/RecBole),
such as (i) dynamically integrating the GNNs with an augmentation model, instead of manually writing a perturbed version, (ii)
adopting a trainer that extends a PyTorch module to perform the augmentation process, (iii) using a specific interface
to manage the sampling policies and the possibility to add a sampling policy as a mere class method applied
on the user or item set.

Most of the models are taken from the Recbole sister library, named [Recbole-GNN](https://github.com/RUCAIBox/RecBole-GNN/tree/main),
while other ones, namely AutoCF, GFCF, SVD-GCN, UltraGCN, were implemented according to their original repository.

# Requirements
Our framework was tested on Python 3.9 with the libraries listed in the
[requirements.txt](FA4GCF/requirements.txt) that can be installed with:
```bash
pip install -r FA4GCF/requirements.txt
```
However, some dependencies, e.g. torch-scatter, could be hard to retrieve
directly from pip depending on the PyTorch and CUDA version you are using, so you should
specify the PyTorch FTP link storing the right libraries versions.
For instance to install the right version of torch-scatter for PyTorch 2.1.2
you should use the following command:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+${CUDA}.html
```
where `${CUDA}` should be replaced by either `cpu`, `cu***`, where `***` represents the
CUDA version, e.g. 117, 121.
__NOTE!__: several models rely on the cuda implementations of some operations provided by torch-scatter and
torch-geometric. We do not guarantee FA4GCF will work on CPU.

# Datasets [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10625046.svg)](https://doi.org/10.5281/zenodo.10625046)


The datasets used in our experiments are Foursquare New York City (FNYC),
Foursquare Tokyo (FKTY), MovieLens 1M (ML1M), Last.FM 1K (LF1K), Rent The Runway (RENT) and
can be downloaded from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10625045).
They should be placed in a folder named _dataset_ in the project root folder,
so next to the _config_ and _FA4GCF_ folders, e.g.:
```
|-- config/*
|-- dataset |-- ml-1m
            |-- lastfm-1k
|-- fa4gcf/*
```

# Usage

The file [main.py](fa4gcf/main.py) is the entry point for every step to execute our pipeline.

## 1. Configuration

FA4GCF scripts are based on similar Recbole config files that can be found in the
-[config](config) folder, with specifics files for each dataset, model, and the augmentation process of each dataset.
Config files are sequentially handled, so if multiple config files contain the same parameters, the used values pertain
to the last config file that appear in the sequence.

The description of the parameters used to train the base model can be found in the Recbole repository
and website, except for this part:
```yaml
eval_args:
    split: {'LRS': None}
    order: RO  # not relevant
    group_by: '-'
    mode: 'full'
```
where `LRS` (Load Ready Splits) is not a Recbole split type, but it is added in our
extended Dataset class to support custom data splits.

The description of each parameter in the __explainer__ config type can be found in the respective files.
In particular, some parameters are related to the original study, and we report them for simplicity:
- `__force_removed_edges__: False` prevents the deletion of a previously added edge. Not used
- `edge_additions: True` => edges are added, not removed
- `exp_rec_data: valid` => the ground truth labels of the validation set are used to measure the approximated NDCG
- `__only_adv_group__: global` => fairness is measured w.r.t the advantaged group utility reported by the base model
- `__perturb_adv_group__: False` >= the group to be perturbed. False perturbs the edges connecting users of
  the disadvantaged group
- `__group_deletion_constraint__: True`: runs the augmentation solely on the disadvantaged group if
  `__perturb_adv_group__` is `False`, otherwise the advantaged group
- `__gradient_deactivation_constraint__: True`: deactivates the gradient computation of the group not targeted by
  the augmentation process. Not used when `__only_adv_group__` is `global`.
- `__random_perturbation__: False`. Not used

## 2. Train Recommender System

The recommender systems need first to be trained:
```bash
python -m FA4GCF.main \
--run train \
--model MODEL \
--dataset DATASET \
--config_file_list config/dataset/DATASET.yaml config/model/MODEL.yaml
```
where __MODEL__ and __DATASET__ denote the model and dataset for the training experiment.
`--config_file_list` can be omitted and the corresponding config files will be automatically loaded if they exist.
`--use_best_params` is a parameter that can be added to use the optimized hyperparameters for each model and dataset
that can match the format `oonfig/model/MODEL_best.yaml`

## 3. Train the FA4GCF Augmentation Process
```bash
python -m FA4GCF.main \
--run explain \
--model MODEL \
--dataset DATASET \
--config_file_list config/dataset/DATASET.yaml config/MODEL/MODEL.yaml \
--explainer_config_file config/explainer/DATASET_explainer.yaml \
--model_file saved/MODEL_FILE
```
where __MODEL_FILE__ is the path of the trained MODEL on DATASET. It can be omitted if the default path `saved` is used.
The algorithm will create a folder with a format similar to
_FA4GCF/experiments/dp_explanations/DATASET/MODEL/PerturbationTrainer/LOSS_TYPE/SENSITIVE_ATTRIBUTE/epochs_EPOCHS/CONF_ID_
where __SENSITIVE_ATTRIBUTE__ can be one of [gender, age], __EPOCHS__ is the number of
epochs used to train FA4GCF, __CONF_ID__ is the configuration/run ID of the run experiment trial.
The folder contains the updated config file specified in `--explainer_config_file` in yaml and pkl format,
a file _cf_data.pkl_ containing the information about the edges added to the graph for each epoch,
a file _model_rec_test_preds.pkl_ containing the original recommendations on the validation set and
test set, a file _users_order.pkl_ containing the users ids in the order _model_rec_test_preds.pkl_ are sorted,
a file _checkpoint.pth_ containing data used to resume the training if stopped earlier.

_cf_data.pkl_ file contains a list of lists where each inner list has 5 values, relative to the edges
added to the graph at a certain epoch:
1) FA4GCF total loss
2) FA4GCF distance loss
3) FA4GCF fair loss
4) fairness measured with the __fair_metric__ (absolute difference of NDCG)
5) the added edges in a 2xN array, where the first row contains the user ids,
the second the item ids, such that each one of the N columns is a new edge of the augmented graph
6) epoch relative to the generated explanations

## 4. Re-Train a Recommender system with an augmented graph
```bash
python -m FA4GCF.main \
--run train \
--model MODEL \
--dataset DATASET \
--config_file_list config/dataset/DATASET.yaml config/model/MODEL.yaml \
--use_perturbed_graph \
--best_exp fairest \
--explanations_path AUGMENTED_GRAPH_PATH
```
where `--best_exp fairest` specifies that the augmented graph should include the edges that reported the fairest
recommendations, which is based on the early stopping criterion (if the patience is 5, the augmented graph banally
pertains to the edges at the position -5 of _cf_data.pkl_, i.e. `cf_data[-5]`), __AUGMENTED_GRAPH_PATH__ is the path
that was created by FA4GCF at step 3, from which the augmented graph is built and used to train the chosen model.
By default, the process saves the model in the __AUGMENTED_GRAPH_PATH__ with the name format of the base model.

# Plotting

The scripts inside the folder [scripts](scripts) can be used to plot the results used in the paper.
