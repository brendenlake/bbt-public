# Bayesian distillation with Behavioral Tuning (BBT)

Researchers must often choose between Bayesian or neural network models of behavior, two paradigms with complementary strengths and weaknesses. An ideal paradigm would facilitate testing many kinds of representations and inductive biases; Bayesian models make this easy, while neural networks do not. Similarly, an ideal paradigm would avoid over-simplifications; neural networks make this easy, while Bayesian models do not. Here, we introduce Bayesian distillation with Behavioral Tuning (BBT) as an approach to getting the best of both traditions. BBT offers a simple recipe for model building: first, a neural network is trained to mimic a Bayesian model through synthetic data, and second, the network is fine-tuned on human behavior to capture additional structure and nuance. Across four case studies in human concept learning, we find that BBT outperforms traditional approaches at predicting human behavior while also revealing psychological insights, resulting in models that can both mimic Bayesian priors and capture heuristics and biases that violate simple modeling assumptions.

Please cite the following article:
[Lake, B. M., Jagadish, A. K., and Jiang, G. (2026). More accurate behavioral predictions with hybrid Bayesian-connectionist models.](https://arxiv.org/abs/2608.22154) Preprint available on arxiv:2608.22154.

## Getting started

Run all commands from the repository root. You should download a directory `data/` from this [link](https://www.dropbox.com/scl/fi/yccrw754po2fctc743uls/bbt_data.zip?rlkey=zb2xvi84x5hhzg3rq80vlwjvi&dl=0) and a directory `out_models/` from this [link](https://www.dropbox.com/scl/fi/os1efy3jmfwj0a8lpg807/bbt_out_models.zip?rlkey=cx1cxj7uwvfkhekeecbnkanuh&dl=0) and place them in the repository root. Note that the checkpoints zip is 3 GB. You can evaluate the provided checkpoints without retraining or regenerating datasets.

Install the core dependencies using Python 3:
```bash
python -m pip install -r requirements.txt
```

The main scripts automatically choose CUDA when available, then Apple Metal (MPS), and otherwise CPU, through `train_utils.get_best_device()`. Training the approximately 30-million-parameter networks can be computationally demanding and was done on a NVIDIA L40 or H200 GPU for the reported results.

For a first evaluation, use the released number-concept BBT checkpoint:

```bash
python eval.py --fn_out_model net_ngame_30M_bbt_top.pt --episode_type number_game_human_test --ll --ll_p_lapse 0.06 --known_length
```

This loads `out_models/net_ngame_30M_bbt_top.pt` and scores the human test responses. The expected reference result appears in the evaluation section below.

Logical-concept Bayesian analysis, plotting, and dataset generation also require **LOTlib3**, which is not bundled here. Follow the [LOTlib setup instructions](#lotlib-dependency) below. Core neural evaluation and training on the existing episode files, as well as number-game dataset generation, work without LOTlib3. Logical-concept plotting (`make_fig4B.py`) and PCA analysis require it.

To inspect the available command-line arguments:

```bash
python train.py --help
python eval.py --help
```

## Repository guide

| File or directory | Purpose |
| --- | --- |
| `train.py` | Distillation and behavioral fine-tuning. |
| `eval.py` | Log-likelihood evaluation, sampled or greedy predictions, internal representations, and compositional-rule HTML output. |
| `datasets.py` | Episode readers, token vocabularies, batching, and task/split selection. |
| `model.py` | Transformer model definitions. |
| `bayes_num_game.py` | Bayesian model of the number game. |
| `bayes_boolean_concepts.py`, `bayes_fol_concepts.py` | Baysesian model of logical concept learning based on LOTlib3. |
| `generate_datasets.py` | Functions for creating synthetic and visualization episode files. |
| `make_fig3B.py`, `make_fig4B.py` | Comparisons of human responses and model predictions for number and logical concepts. |
| `sparse_pp_approx.py` | Sparse approximations of posterior-predictive distributions for model interpretation. |
| `data/` | Synthetic episodes, human responses, and visualization inputs. |
| `out_models/` | Released checkpoints and the default destination for training outputs. |
| `analysis_sys_gen/` | Compositional-rule visualization templates, assets, and analysis code. |

### Data and episode types

Each learning episode pairs a **support set** of examples with **query inputs and outputs**. Synthetic query outputs train the network during distillation; human query responses are used during behavioral fine-tuning and evaluation. Text episode files contain `*SUPPORT*`, `*QUERY*`, and `*HYPOTHESIS*` (or `*GRAMMAR*`) sections, with example lines of the form `IN: ... OUT: ...`. `datasets.py` converts these files into model inputs and targets.

| Case study | Synthetic distillation | Human fine-tuning | Held-out human evaluation | Main data locations |
| --- | --- | --- | --- | --- |
| Number concepts | `number_game` | `number_game_human` | `number_game_human_test` | `data/num_game/`, `data/num_game_viz/` |
| Logical concepts | `set_concepts_uniform` | `set_concepts_human_ftune` | `set_concepts_human_test` | `data/set_concepts_uniform/`, `data/set_concepts/`, `data/set_concepts_viz/` |
| Compositional rules | `sys_gen` | `sys_gen_alltune_human` | `sys_gen_test_human` | `data/sys_gen/data_algebraic/`, `data/sys_gen/data_human/` |

### Released checkpoints

`out_models/` contains three checkpoint families for each task:

| Task prefix | Case study |
| --- | --- |
| `net_ngame_30M` | Number concepts |
| `net_set_30M` | Logical concepts |
| `net_sysgen_30M` | Compositional rules |

The suffix `*_bbt_top.pt` identifies the released behaviorally tuned model, `*_base_paired.pt` its paired distillation-only model, and `*_scratch_top.pt` the behavioral model trained from scratch. To compare these models, substitute the corresponding filename in a task's evaluation command while keeping all scoring options fixed.

## Evaluation

The commands below evaluate the released BBT checkpoints on held-out human responses. Their reference outputs correspond to Table 1; small numerical differences can arise from hardware and dependency versions. These are reference results, not a guarantee of identical output on every system.

`eval.py` loads models from `out_models/` by default. Despite its name, `--fn_out_model` identifies the checkpoint to **load** during evaluation. Use `--dir_model` to choose another checkpoint directory. The checkpoint supplies the network dimensions and, unless overridden, the batch size.

Key evaluation options:

| Option | Meaning |
| --- | --- |
| `--episode_type` | Selects the dataset and split; the mappings are defined in `datasets.get_dataset`. |
| `--ll` | Computes total log-likelihood and the number of predicted symbols. Higher (less negative) log-likelihood is better when comparing the same data and scoring settings. |
| `--ll_p_lapse` | Mixes the network's predictions with a uniform lapse distribution, using the specified probability. |
| `--known_length` | Scores output symbols without predicting the end-of-sequence token. |
| `--ll_nrep` | Repeats evaluation over stochastic transformations and reports aggregate statistics. |
| `--batch_size` | Overrides the checkpoint's batch size; reducing it can lower memory use. |

### Evaluating BBT on number concepts

To evaluate the log-likelihood of the best run:

```bash
python eval.py --fn_out_model net_ngame_30M_bbt_top.pt --episode_type number_game_human_test --ll --ll_p_lapse 0.06 --known_length
```

This should print numbers like the following:
`evaluation on number_game_human_test loglike: -28157.9 for 54480 symbol predictions`

To make the generalization plots in Fig. 3B and 3C (saved in a folder `figs_ngame_trials_test`):

```bash
python make_fig3B.py
```

To make PCA plots in Fig. 3D (PDFs in `figs_embedding/`, with `ModelInternal-*.pkl` caches in the repository root):

```bash
python eval.py --fn_out_model net_ngame_30M_base_paired.pt --get_internal_num_game --episode_type number_game_viz --batch_size 1
python eval.py --fn_out_model net_ngame_30M_bbt_top.pt --get_internal_num_game --episode_type number_game_viz --batch_size 1
```

### Evaluating BBT on logical concepts

To evaluate the log-likelihood of the best run:

```bash
python eval.py --fn_out_model net_set_30M_bbt_top.pt --episode_type set_concepts_human_test --ll --ll_p_lapse 0.04 --batch_size 10 --known_length
```

This should print numbers like the following:
`evaluation on set_concepts_human_test loglike: -69423.8 for 179965 symbol predictions`

To make the logical-concept generalization plots associated with Fig. 4C (saved in a folder `figs_set_trials_test`):

```bash
python make_fig4B.py
```

To make PCA plots in Fig. 4D (PDFs in `figs_embedding/`, with `ModelInternal-*.pkl` caches in the repository root; requires LOTlib3):

```bash
python eval.py --episode_type set_concepts_viz --fn_out_model net_set_30M_base_paired.pt --get_internal_set_concepts --batch_size 1
python eval.py --episode_type set_concepts_viz --fn_out_model net_set_30M_bbt_top.pt --get_internal_set_concepts --batch_size 1
```

### Evaluating BBT on Shepard concepts

Commands and checkpoints for the Shepard-concepts case study are not included in this release. The workflows below cover the other three case studies.

### Evaluating BBT on compositional rules

To evaluate the log-likelihood of the best run:

```bash
python eval.py --fn_out_model net_sysgen_30M_bbt_top.pt --ll --ll_nrep 100 --batch_size 10 --episode_type sys_gen_test_human
```

This should print numbers like the following:
`  loglike: M = -343.3 (SD= 11.7996 , Nrep= 100 ) for 990.0 (SD= 0.0 ) symbol predictions on average`

To visualize the model outputs as in Fig. 6B (saved in folder `analysis_sys_gen/sys_gen_test_human_mult10`):

```bash
python eval.py --episode_type sys_gen_test_human_mult10 --fn_out_model net_sysgen_30M_bbt_top.pt --sys_gen_html --batch_size 5
python eval.py --episode_type sys_gen_test_human_mult10 --fn_out_model net_sysgen_30M_base_paired.pt --sys_gen_html --batch_size 5
```

## Training

BBT training has two stages. First, a neural network is trained on synthetic episodes sampled from a Bayesian generative model. Second, the network is fine-tuned on human responses so that its predictions can capture behavioral structure beyond the original Bayesian model.

The commands below train new models using the included datasets. They do not overwrite the released `*_base_paired.pt` or `*_bbt_top.pt` checkpoints. The architecture arguments specify four encoder layers, four decoder layers, and 512-dimensional embeddings, matching the approximately 30-million-parameter configuration used here.

Training saves checkpoints under `out_models/` by default. With `--save_best`, it also writes a separate `*_best.pt` checkpoint selected by validation loss; the ordinary `.pt` file contains the latest saved state. For example, number-concept fine-tuning writes `net_ngame_30M_ftune_best.pt` as well as `net_ngame_30M_ftune.pt`. Use the best checkpoint when evaluating the validation-selected model.

To resume an interrupted run, repeat its original command with `--resume` and the same output filename and settings; `--resume` cannot be combined with `--fn_in_model`.

### Training BBT on number concepts

Distilling the Bayesian model:

```bash
python -u train.py --grad_clip 1.0 --lr 0.0001 --patience 4 --episode_type number_game --fn_out_model net_ngame_30M_base.pt --nepochs 50 --batch_size 50 --nlayers_encoder 4 --nlayers_decoder 4 --emb_size 512
```

Fine-tuning the model on human behavior:

```bash
python -u train.py --grad_clip 1.0 --lr 0.0001 --episode_type number_game_human --fn_in_model net_ngame_30M_base.pt --fn_out_model net_ngame_30M_ftune.pt --save_best --nepochs 20 --batch_size 25 --nlayers_encoder 4 --nlayers_decoder 4 --emb_size 512
```

### Training BBT on logical concepts

Distilling the Bayesian model:

```bash
python -u train.py --grad_clip 1.0 --lr 0.0001 --patience 5 --episode_type set_concepts_uniform --fn_out_model net_set_30M_base.pt --nepochs 50 --batch_size 100 --nlayers_encoder 4 --nlayers_decoder 4 --emb_size 512
```

Fine-tuning the model on human behavior:

```bash
python -u train.py --grad_clip 1.0 --lr 0.0001 --episode_type set_concepts_human_ftune --fn_in_model net_set_30M_base.pt --fn_out_model net_set_30M_ftune.pt --save_best --nepochs 20 --batch_size 10 --nlayers_encoder 4 --nlayers_decoder 4 --emb_size 512
```

### Training BBT on compositional rules

Distilling the Bayesian model:

```bash
python -u train.py --grad_clip 1.0 --lr 0.0001 --patience 4 --episode_type sys_gen --fn_out_model net_sysgen_30M_base.pt --nepochs 50 --batch_size 50 --nlayers_encoder 4 --nlayers_decoder 4 --emb_size 512
```

Fine-tuning the model on all 8 grammars of human behavior (Note that the article had more complex procedure for finding the number of fine-tuning epochs through cross-validation):

```bash
python -u train.py --grad_clip 1.0 --lr 0.0001 --no_lr_schedule --episode_type sys_gen_alltune_human --fn_in_model net_sysgen_30M_base.pt --fn_out_model net_sysgen_30M_allftune.pt --nepochs 8 --batch_size 10 --nlayers_encoder 4 --nlayers_decoder 4 --emb_size 512
```

## Adding your own Bayesian model

1. **Generate synthetic episodes.** Sample a hypothesis from your model's prior, then sample support and query examples conditional on that hypothesis. This requires forward sampling rather than posterior inference. Follow `generate_num_game_dataset` in `generate_datasets.py`, saving tokenized examples with `episode_to_file`.
2. **Add a dataset loader.** Adapt a class in `datasets.py` to return episodes through `bundle_bbt_episode`, define input/output vocabularies in `langs` and a maximum support-set size in `max_ns`, and register synthetic and human train/validation/test splits in `get_dataset`. Keep token mappings identical across synthetic and human data.
3. **Distill, then tune.** Adapt the training commands above to your episode types: first train on synthetic data, then use `--fn_in_model` to fine-tune that checkpoint on human responses with the same architecture. Select settings on validation data and evaluate the final model on held-out human responses using `eval.py --ll`. For tokenized input/output tasks, the existing `BBT` network can generally be reused.

## LOTlib dependency

Install [LOTlib3](https://github.com/piantado/LOTlib3) separately from Steven Piantadosi for logical-concept Bayesian analyses, plotting, PCA, or dataset generation. Installing `requirements.txt` does not supply it.

Clone the library (skip this step if you already have a checkout):

```bash
mkdir -p "$HOME/Libraries"
git clone https://github.com/piantado/LOTlib3.git "$HOME/Libraries/LOTlib3"
```

Add the **parent directory** of `LOTlib3` to Python's import path in the terminal where you run the analyses:

```bash
export PYTHONPATH="$HOME/Libraries${PYTHONPATH:+:$PYTHONPATH}"
python -c "from LOTlib3.Grammar import Grammar; import bayes_fol_concepts; print('LOTlib3 is ready')"
```

Run the check from this repository's root using the same Python environment as your analyses. If you cloned LOTlib3 elsewhere, replace `$HOME/Libraries` with its parent directory. To keep the setting across terminal sessions, add the `export` line to your shell configuration (for example, `~/.zshrc` or `~/.bashrc`). The Bayesian modules report these setup requirements if LOTlib3 is missing; they do not modify Python's import path.