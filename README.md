# Reward Hacking Detection via Unsupervised Anomaly Detection

## Overview

Reward hacking occurs when an AI agent exploits flaws or ambiguities in its reward function to achieve high scores without actually solving the intended task. In coding agents, this can include modifying unit tests instead of fixing code, bypassing runtime constraints, or hardcoding outputs for known test cases. Detecting this behavior automatically and reliably remains an open problem in AI safety.

This project studies reward hacking detection as an unsupervised anomaly detection problem. Instead of training on labeled hacking examples, the models are trained only on benign coding-agent trajectories and evaluated on whether hacked trajectories look anomalous relative to normal behavior.

Two methods are compared:

- Variational Autoencoder (VAE): learns a latent representation of benign trajectories; hacked trajectories are expected to have higher reconstruction error and KL divergence.
- Deep SVDD: maps benign trajectories into a compact region in representation space; trajectories farther from that region are treated as anomalies.

Training uses SWE-smith benign coding-agent trajectories. Evaluation uses TRACE, a synthetic benchmark of hacked and benign coding trajectories, and MALT, a real-world dataset containing naturally occurring reward hacking examples. Using both TRACE and MALT allows evaluation on both synthetic and real reward hacking behavior.

## Research Question

Can unsupervised models trained only on benign coding-agent trajectories learn a representation of normal behavior from which reward hacking deviates in a detectable and consistent way across datasets?

This is evaluated through three questions:

1. Do VAE reconstruction error and KL divergence separate hacked from benign trajectories in TRACE and MALT?
2. Does Deep SVDD provide a better anomaly score than the VAE, especially for subtle hacks?
3. Do hacked trajectories from TRACE and MALT occupy similar regions in the learned latent space?

## Datasets

### SWE-smith Trajectories

`huggingface: SWE-bench/SWE-smith-trajectories`

Training dataset of **5,017 benign multi-turn coding-agent trajectories** generated with SWE-agent on real GitHub issues. This dataset is used only for training.

### TRACE

`huggingface: PatronusAI/trace-dataset`

Evaluation dataset of 517 multi-turn coding-agent trajectories, including 268 hacked and 249 benign examples. Only the binary hacked/benign label is used.

### MALT

`huggingface: metr-evals/malt-public`

Evaluation dataset of software and research agent transcripts containing naturally occurring reward hacking behavior. It includes:

- 103 manually reviewed hacked runs
- 1,01* manually reviewed benign runs
- 8,229 presumed benign runs

## Method

Trajectories from all datasets are first normalized into a shared format and converted into fixed-dimensional embeddings using a frozen sentence encoder. Turn-level embeddings are then aggregated into one trajectory-level vector.

The models are trained on SWE-smith benign embeddings only and evaluated on TRACE and MALT without fine-tuning.

A simple baseline is also included: cosine distance from the centroid of benign SWE-smith embeddings in the raw embedding space.

## Goal

The goal is to test whether reward hacking can be detected as a deviation from normal coding-agent behavior without requiring hacked examples during training, and whether that detection generalizes across both synthetic and real-world datasets.