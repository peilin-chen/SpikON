# SpikON: A Dual-Parallel and Efficient Accelerator for Online Spiking Neural Networks Learning
This repository contains the code asscoiated with "SpikON: A Dual-Parallel and Efficient Accelerator for Online Spiking Neural Networks Learning", accepted to ISLPED2026. It contains the algorithm for SpikON.

## Introduction
Spiking neural networks (SNNs) have emerged as a promising paradigm for energy-efficient brain-inspired computing. However, existing online unsupervised SNN learning suffers from low training accuracy and poor scalability. Although current online supervised learning algorithm performs well on large-scale datasets and networks, the non-hardware-friendly operations hinder efficient edge deployment. In this work, we propose SpikON, the first algorithm-hardware co-design framework for efficient and scalable end-to-end online supervised SNN learning. We first propose the learnable threshold through time and scaled weight centralization through time techniques to address the inefficiency of traditional algorithms. Moreover, to reduce latency and energy consumption, we introduce the novel training dataflow and cascade computation reuse scheme for SNNs that allows concurrent forward-backward computation and temporal reuse across timesteps. We further design the dedicated SNN accelerator with a dual-parallel engine and customized SIMD-based SNN core for efficient end-to-end online learning. Experiments show that the SpikON algorithm achieves 32.2% and 35.0% reductions in training latency and energy consumption over the baseline, without sacrificing accuracy. Moreover, the SpikON algorithm-hardware co-design achieves 7.2x (11.5x) and 26.8x (15.8x) training throughput (energy efficiency) compared with the edge Apple M4 GPU and TPU-like accelerator, respectively.

## Run Algorithm
Please look at the Jupyter Notebooks.

## Citation
to be updated

## Reference Repositories
SLTT: https://github.com/qymeng94/SLTT