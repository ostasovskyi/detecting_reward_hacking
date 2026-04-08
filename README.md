# Reward Hacking Detection via Unsupervised Anomaly Detection

## Overview

Reward hacking occurs when an AI agent exploits flaws or ambiguities in its reward function to achieve high scores without actually solving the intended task. In coding agents, this can include modifying unit tests instead of fixing code, bypassing runtime constraints, or hardcoding outputs for known test cases. Detecting this behavior automatically and reliably remains an open problem in AI safety.

This project studies reward hacking detection as an unsupervised anomaly detection problem. The core premise is that benign coding agent behavior could have different latent structure and hacking behavior deviates from this structure in detectable ways. Instead of training on labeled hacking examples (which realistically don't have access to), the models we will work on are trained only on benign coding-agent trajectories and evaluated on whether hacked trajectories look anomalous relative to normal behavior.

Two methods are compared:

- Variational Autoencoder (VAE): learns a probabilistic latent representation of benign trajectories, so hacked trajectories are expected to have higher reconstruction error and higher KL divergence from the prior;
- Deep Support Vector
Data Description (Deep SVDD) from Ruff et al., ICML 2018: train a neural network to map benign trajectories into a compact rhypersphere in representation space, with trajectories farther from that region are treated as anomalies.


The motivation for trying both is that VAE-based detection can fail when hacked trajectories are surface-plausible but structurally-anomalous, because the reconstruction objective is sensitive to similarity rather than deep behavioral structure. Deep SVDD has no reconstruction objective and is instead optimized directly for compactness of the normal distribution, making it potentially more robust to subtle hacks that have been shown to be the hardest to detect. 


## Research Question


The goal is to test whether reward hacking can be detected as a deviation from normal coding-agent behavior without requiring hacked examples during training, and whether that detection generalizes across both synthetic and real-world datasets. 

Therefore, we ask if  unsupervised models trained only on benign coding-agent trajectories learn a representation of normal behavior from which reward hacking deviates in a detectable and consistent way. This is evaluated through two questions:

1. Do VAE reconstruction error and KL divergence separate hacked from benign trajectories in TRACE and MALT? Does Deep SVDD provide a better anomaly score than the VAE? 
2. Do hacked trajectories from TRACE (synthetically generated coding trajectories) and MALT (real-world codind trajectories) occupy similar regions in the learned latent space?


## Datasets

#### SWE-smith Trajectories

`huggingface: SWE-bench/SWE-smith-trajectories`

Training dataset of **5,017 benign multi-turn coding-agent trajectories** generated with SWE-agent on real GitHub issues. This dataset is used only for training.

#### TRACE

`huggingface: PatronusAI/trace-dataset`

Evaluation dataset of 517 multi-turn coding-agent trajectories, including **268 hacked and 249 benign syntethic examples**. Only the binary hacked/benign label is used.

#### MALT

`huggingface: metr-evals/malt-public`

Evaluation dataset of software and research agent transcripts containing naturally occurring reward hacking behavior. It includes:

- **103 manually reviewed hacked runs**
- **1,01 manually reviewed benign runs**
- 8,229 presumed benign runs


## Method

Trajectories from all datasets are first normalized into a shared format and converted into fixed-dimensional embeddings using a frozen sentence encoder. Turn-level embeddings are then aggregated into one trajectory-level vector. The models are trained on SWE-smith benign embeddings only and evaluated on TRACE and MALT without fine-tuning.

Before the VAE and Deep SVDD, we include simple baseline via cosine distance from the centroid of benign SWE-smith embeddings in the raw embedding space.  This tests whether the pretrained encoder alone already separates hacked from benign, without any task-specific training.

The argument for the VAE and Deep SVDD is that the frozen encoder was not trained to distinguish coding agent behavior from hacking behavior, so the embedding space may not organize along that axis at all. However, doing some training on these trajectories could in principle learn a lower-dimensional latent space that captures the specific dimensions of variation within normal coding agent behavior, making deviations more detectable. This is our research hypothesis, not a guarantee!

