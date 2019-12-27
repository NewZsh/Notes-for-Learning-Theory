# Notes of Statistical Machine Learning Theory

*The notes is mainly based on the book **Understanding Machine Learning: From Theory to Algorithms** XXX, 2014?*

## 1. Probably Approximately Correct (PAC)

*Corresponding to Chapter 2-5 in UML.*
*This part mainly answers the quesion: What can we know about the generalization error? How does the hypothesis set (in application, the choice of classifier, regressor or so on) reflect our prior knowledge, or, inductive bias?*

### 1.1 formulation

**The learner's input**:

- Domain Set: instance $x \in \mathcal{X}$.
- Label Set: label $y \in \mathcal{Y}$. Currently, just consider the binary classification task, i.e., $y = 0,1 $ or $ -1, +1 $.
- Training data: $S=((x_1, y_1), \cdots, (x_m,y_m))$ is a finite sequence.

        remark: usually called 'training set', but must be 'training sequence', because the same sample may appear more than one time, and some training algorithms is order-sensitive.

  - **A simple data generation model**:

  
**The learner's output**: hypothesis (or classifier, regressor) $h: \mathcal{X}\rightarrow\mathcal{Y}$.

**Evaluation**: *a.k.a* **generalization error**, or true error/risk

$$
L_{\mathcal{D},f}(h) \overset{def}{=} \mathop{\mathbb{P}}\limits_{x \sim \mathcal{D}} [h(x) \neq f(x)] \overset{def}{=} \mathcal{D}(\{ x:h(x) \neq f(x)\} )
$$

  **[NOTE]**: by the subscript $\mathcal{D}, f$, it means that the error of $h$ is the probability to draw a random instance $x$, according to the distribution \mathcal{D}, such that $h(x)\neq f(x)$.

        remark: here we neglect the measurability assumption.

### 1.2 Empirical Risk Minimization (ERM)

- ERM may overfit
- ERM with restricted hypothesis set: inductive bias is introduced
- No-Free-Lunch: inductive bias is neccessary
  
### 1.3 PAC and Agnostic PAC

- Realizability assumption and finite classes
- Beyond realizability assumption: Agnostic PAC
- Beyond binary classification: learning via uniform convergence

### 1.4 Error decomposition

*Now that, we have come to some important conclusions under the PAC learning framework:*
*1. No universal learner;*
*2. inductive bias is neccessary to avoid overfitting;*
*3. sample complexity is function about hypothesis set, confidence level and error, interestingly, it is nothing to do with the dimension of feature space;*
*4. inductive bias controls the balance of approximation error and estimation error.*

### 1.4 Excercises and solutions

#### 1.4.1 (UML Ex2.1)
solution

#### 1.4.2 (UML)