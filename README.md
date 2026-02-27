# The Brittleness of Efficiency: Adversarial Sensitivity in Compressed Vision Models

**A research investigation into whether model compression via pruning makes neural networks more vulnerable to adversarial attacks.**

### Overview
Modern neural networks are frequently compressed (e.g., via pruning) to enable efficient deployment on edge devices, supporting sustainable and low-carbon AI. However, does this efficiency come at the hidden cost of reduced adversarial robustness?

This project explores the trade-off by applying unstructured pruning to a pre-trained ResNet-18 on CIFAR-10 and evaluating its sensitivity to Fast Gradient Sign Method (FGSM) attacks. Integrated Gradients visualizations reveal how pruning shifts attention to brittle features.

Key motivation: Naive compression may introduce reliability risks in safety-critical or resource-constrained applications.

### Key Findings (from a 2-hour Colab experiment)
- **Robustness Cliff Observed**: At 75% sparsity, attack success rate (ASR) under FGSM (ε=8/255) increased by ~30–45% compared to the baseline (e.g., from ~60–70% to near 100% on many examples), while clean top-1 accuracy dropped only ~3–8%.
- **Attention Drift**: Integrated Gradients attribution maps show pruned models increasingly rely on high-frequency noise and non-semantic patterns rather than robust, meaningful features — providing visual evidence of lost inductive bias.
- **Implication**: Aggressive pruning without robustness-aware techniques can make models significantly more fragile to simple adversarial perturbations, highlighting a need for careful design in efficient AI systems.

These results align with prior observations in the literature (e.g., increased vulnerability in high-compression regimes without adversarial fine-tuning).

### Quick Setup & Reproduction
1. Open the notebook in Google Colab (free GPU recommended):  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/brittleness-of-efficiency/blob/main/notebook/Brittleness_of_Efficiency.ipynb)

2. Install dependencies (one cell):  
   !pip install foolbox captum torch torchvision

3. Run all cells sequentially — baseline → pruning → FGSM attacks → Integrated Gradients viz.

### Technologies Used
1. PyTorch (ResNet-18 pre-trained on CIFAR-10)
2. Foolbox (for standardized FGSM attacks)
3. Captum (Integrated Gradients for interpretability)
4. Matplotlib / Seaborn (visualizations)

### Limitations & Future Directions
No post-pruning fine-tuning (findings reflect naive compression effects)
Single attack (FGSM); stronger iterative attacks (e.g., PGD) likely amplify the cliff
CIFAR-10 only; extend to ImageNet or real-world datasets
Explore robustness-aware pruning or quantization for mitigation
