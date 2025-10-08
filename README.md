# Machine-learning-MECs

A machine learning–based workflow for predicting and optimizing multi-element catalyst (MEC) performance through a combination of neural networks (MLP) and genetic algorithms (GA).   
The GA generates new candidate compositions based on fitness values predicted by the trained MLP model, enabling iterative optimization toward optimal catalyst performance.  
This repository provides the full code for model training, GA-guided optimization, and performance evaluation.To interpret model predictions, SHAP (SHapley Additive exPlanations) 
was employed to quantify the contribution of each element (Ni, Co, Fe, Pd, Pt) to the predicted catalytic performance.

---

## 1. Create a conda or virtual environment

conda create -n mec_ml python=3.9  
conda activate mec_ml


## 2. Install dependencies

numpy pandas scikit-learn torch matplotlib shap  

## 3. How to Run
### 3.1. GA searching process
```bash
python ga_main.py
```

### 3.2. Train MLP model
```bash
python mlp_main.py
```

### 3.3. GA-mlp closed-loop optimization
```bash
python ga_integrated_mlp.py
```
