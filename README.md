# Quantum-Connectivity-for-Quantum-Networks

This repository contains the Python codes used to reproduce the numerical results presented in:

**"Quantum connectivity of quantum networks", Md Sohel Mondal, Shashank Shekhar, and Siddhartha Santra**

📄 **arXiv:** https://arxiv.org/abs/2603.29601

The paper introduces the **Quantum Connectivity Measure (QCM)**, **Quantum-Connected Fraction (QCF)**, and **Quantum Clustering Coefficient (QCC)** to characterize the functional connectivity of quantum networks beyond their underlying physical topology.

## Contents

| File | Figure | Description |
|---|---|---|
| `qcm_and_qcf_vs_conc.py` | Fig. 2(a) | QCM and QCF vs. average edge concurrence for fully connected and random networks. |
| `qcm_waxman_network.py` | Fig. 2(b) | Regional QCM for a spatial Waxman quantum network. |
| `path_length_pdf.py` | Supplemental Fig. 1 | Shortest-path-length distribution for a random network. |

## Requirements

Python 3 with:

```bash
pip install numpy scipy matplotlib networkx
