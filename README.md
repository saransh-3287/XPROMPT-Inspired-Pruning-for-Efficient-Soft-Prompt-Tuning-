#  XPROMPT-Inspired Soft Prompt Pruning using BART-base on SuperGLUE Tasks

##  1. Aim
This project implements an **XPROMPT-inspired framework** for efficient **soft prompt tuning** on five **SuperGLUE benchmark** tasks:
> **COPA, WiC, WSC, CB, and RTE**

We use the **BART-base** model to explore whether **pruning redundant or uninformative soft prompt tokens** can maintain or improve downstream task performance **while reducing tunable parameters**.

---

##  2. Methodology

### 2.1 Model Setup
- **Base Model:** `facebook/bart-base` (Hugging Face)
- **Trained Parameters:** Only the *soft prompt embeddings* (20 tokens × 1024 dimensions)
- **Frozen Parameters:** All BART model parameters
- **Optimizer:** Adafactor  
  - Learning rate: `0.3`  
  - Weight decay: `1 × 10⁻⁵`
- **Training Schedule:**  
  - 90 epochs for initial soft prompt tuning  
  - +10 epochs for retraining after each pruning configuration

---

### 2.2 XPROMPT Framework
The pruning process follows the four-step **XPROMPT** approach:

1. **Soft Prompt Tuning**  
   - Learnable prompt embeddings prepended to input embeddings.  
   - Tuned using each downstream dataset.

2. **Token-Level Pruning**  
   - Compute token importance via **gradient magnitudes**.  
   - Remove lowest *r%* tokens (least important).

3. **Piece-Level Pruning**  
   - Split each token embedding into 16 equal parts (dimension-wise).  
   - Compute piece-level importance; prune less informative segments.

4. **Rewinding & Retraining**  
   - Restore surviving embeddings to pre-pruned values.  
   - Retrain with frozen BART parameters for stability.

5. **Evaluation**  
   - Compare validation accuracy before/after pruning.  
   - Measure retained tunable parameters.

---

##  3. Experimental Setup

| **Hyperparameter** | **Value** |
|--------------------|-----------|
| Prompt Length | 20 |
| Embedding Dimension | 1024 |
| Piece Splits | 16 |
| Optimizer | Adafactor |
| Learning Rate | 0.3 |
| Weight Decay | 1×10⁻⁵ |
| Epochs | 90 + 10 (per pruning rate) |
| Batch Size | 16 |
| Max Input Length | 256 |
| Max Output Length | 8 |
| Pruning Ratios | [0.1, 0.3] |

---

##  4. Results

###  4.1 COPA
| Metric | Value |
|---------|--------|
| Accuracy (Before Pruning) | 45.00% |
| Accuracy (After Pruning) | **55.00%** |
| Best Ratio | Token 30%, Piece 10% |
| Parameters Retained | ~61% |

**Observation:**  
Moderate pruning improved performance by **+10 percentage points**, acting as a regularizer by removing noisy prompt tokens.

---

###  4.2 WiC
| Metric | Value |
|---------|--------|
| Accuracy (Before Pruning) | 53.13% |
| Accuracy (After Pruning) | **56.90%** |
| Best Ratio | Token 10%, Piece 10% |
| Parameters Retained | ~79% |

**Observation:**  
Mild pruning (10%) improved accuracy slightly, indicating some tokens were redundant.

---

###  4.3 WSC
| Metric | Value |
|---------|--------|
| Accuracy (Before Pruning) | 40.38% |
| Accuracy (After Pruning) | **63.46%** |
| Best Ratio | Token 10%, Piece 10% |
| Parameters Retained | ~79% |

**Observation:**  
Significant gain (**+23 pp**). Pruning clarified representational focus by removing misleading tokens.

---

###  4.4 CB
| Metric | Value |
|---------|--------|
| Accuracy (Before Pruning) | 67.86% |
| Accuracy (After Pruning) | **71.43%** |
| Best Ratio | Token 10%, Piece 10% |
| Parameters Retained | ~79% |

**Observation:**  
Performance remained strong; minimal pruning preserved discriminative ability.

---

###  4.5 RTE
| Metric | Value |
|---------|--------|
| Accuracy (Before Pruning) | 48.01% |
| Accuracy (After Pruning) | **54.87%** |
| Best Ratio | Token 30%, Piece 10% |
| Parameters Retained | ~61% |

**Observation:**  
Moderate pruning improved validation accuracy by **+6.8 pp**, efficiently removing redundant prompt parts.

---

##  5. Visualization Summary
Each task included:
-  **Bar plots** of token-level gradient importance  
-  **Binary token retention plots** (kept vs pruned tokens)  
-  **Heatmaps** of piece-level importance  
-  **Piece retention maps** showing pruning density

---

##  6. Discussion: Effect of Pruning
Across all **five SuperGLUE tasks**, pruning improved the **efficiency–performance trade-off**.

- Uninformative or noisy soft-prompt tokens were removed using **gradient-based importance**.
- Resulted in **21–39% reduction** in tunable parameters.
- Validation accuracy **matched or exceeded** full soft-prompt tuning after pruning and retraining.
- Significant improvements for smaller datasets like:
  - **WSC:** +23 percentage points  
  - **COPA:** +10 percentage points

 **Key Insight:**  
> Pruning acts as a structured regularizer — it prevents overfitting and enhances representation focus while reducing parameters.

Overall, **XPROMPT pruning** produces a *leaner prompt representation* with **better or equal performance** than standard soft-prompt tuning, offering a **superior balance between efficiency and accuracy**.

---

