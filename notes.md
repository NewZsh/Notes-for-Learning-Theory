# Notes of Statistical Machine Learning Theory

*The notes is mainly based on the book **Understanding Machine Learning: From Theory to Algorithms** XXX, 2014?*

## 1. Formulation

### 1.1 Probably Approximately Correct (PAC)

*Corresponding to Chapter 2-5 in UML. This part mainly answers the quesion: What can we know about the generalization error? How does the hypothesis set (in application, the choice of classifier, regressor or so on) reflect our prior knowledge, or, inductive bias?*

**The learner's input**:

- Domain Set: instance $x \in \mathcal{X}$.
  
- Label Set: label $y \in \mathcal{Y}$. Currently, just consider binary classification task, i.e., $y = 0,1 $ or $ -1, +1 $.

- Training data: $S=((x_1, y_1), \cdots, (x_m,y_m))$ is a finite sequence.

<font size=7>remark: usually called 'training set', but must be *'training sequence'* more exactly, because the same sample may appear more than one time, and some training algorithms is order-sensitive.</font>

    - **A simple data generation model**:

**The learner's output**:

**Evaluation**: