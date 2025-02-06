# Adversarial-Robustness-ML

### GitHub Repository Summary: Adversarial Robustness in Machine Learning

This repository contains three Jupyter notebooks developed during my Master's in AI and Data Science at Université Paris Dauphine. The projects focus on **adversarial robustness**, exploring how machine learning models can be vulnerable to carefully crafted perturbations (attacks) and methods to create robust classifiers.

---

### Notebook 1: **Linear Models and Adversarial Examples**  
**Objective**: Demonstrate vulnerability of linear classifiers to adversarial attacks.  
**Methods**:  
- Trained a **binary linear classifier** on MNIST (digits 0 and 1) using PyTorch.  
- Generated adversarial perturbations using the formula:  
  ```python
  delta = eps * model.linear.weight.detach().sign()
  ```  
- Evaluated model performance on natural and adversarial test sets.  

**Key Results**:  
- Test accuracy dropped from **~99%** (natural) to **~35%** under attack.  
- Perturbations aligned with the model’s weight vector, visible as a "1" pattern.  

**Conclusion**:  
Linear models are highly susceptible to adversarial examples. Attacks exploit the model’s linear structure, highlighting the need for non-linear defenses.

---

### Notebook 2: **Neural Networks and Advanced Attacks**  
**Objective**: Extend adversarial attacks to neural networks and real-world images.  
**Methods**:  
- Trained a **CNN** on CIFAR-10 and a pre-trained ResNet50 on ImageNet.  
- Implemented **FGSM** (single-step) and **PGD** (iterative) attacks:  
  ```python
  delta = eps * torch.sign(gradient)  # FGSM
  delta = clip(delta + alpha * gradient)  # PGD (iterative)
  ```  
- Crafted adversarial examples for MNIST/CIFAR-10 and a real-world pig image.  

**Key Results**:  
- FGSM reduced CNN accuracy from **~75%** to **~25%**; PGD further degraded performance.  
- ResNet50 misclassified a pig as a **wombat** with imperceptible perturbations.  

**Analysis**:  
- Iterative attacks (PGD) are more potent than single-step FGSM.  
- Adversarial examples transfer across models, posing real-world risks.  

**Visualization**:  
an Adversarial pig .jpg

---

### Notebook 3: **Defenses and Robust Training**  
**Objective**: Explore methods to improve model robustness.  
**Methods**:  
- (Inferred) Implemented **adversarial training** by augmenting training data with adversarial examples.  
- Evaluated trade-offs between natural accuracy and robustness.  

**Key Insights**:  
- Models trained with adversarial examples show improved resilience to attacks.  
- Robust optimization (e.g., minimizing worst-case loss) is critical for safety-critical applications.  

---

### Synthesis and Contributions  
1. **Progression of Attacks**:  
   - Linear models → CNNs → Real-world images.  
   - Highlighted scalability of adversarial threats.  

2. **Attack Effectiveness**:  
   - **Linear models**: Vulnerable to weight-aligned perturbations.  
   - **CNNs**: Susceptible to iterative gradient-based attacks (PGD).  

3. **Defense Strategies**:  
   - Adversarial training and robustness certification are promising directions.  

**Conclusion**: Adversarial examples reveal fundamental weaknesses in modern ML systems. Future work must balance accuracy and robustness for real-world deployment.

---

### Repository Structure  
```
├── Notebook1_Linear_Adversarial.ipynb  # Binary MNIST, linear attacks  
├── Notebook2_CNN_FGSM_PGD.ipynb       # CIFAR-10/ResNet50, advanced attacks  
└── Notebook3_Robust_Training.ipynb    # Defense mechanisms (adversarial training)  
```

**ML Libs**: PyTorch.  
**Datasets**: MNIST, CIFAR-10, ImageNet.  

