**Notes of Statistical Machine Learning Theory**

*The notes is mainly based on the following book*

- UML: [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)  Shai Shalev-Shwartz and Shai Ben-David, 2014.
- PRML: [pattern recognition and machine learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) Christopher M. Bishop, 2006.
- PGM: [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models) Daphne Koller and Nir Friedman, 2009.
- GEV: [Graphical Models, Exponential Families, and Variational Inference](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf) Martin J. Wainwright and Michael I. Jordan, 2008.

**Chapter TWO VC-dimension**


- [1.1 Finitness is not a necessary condition for learnability](#11-finitness-is-not-a-necessary-condition-for-learnability)
  - [1.1.1 The set of threshold functions is PAC learnable](#111-the-set-of-threshold-functions-is-pac-learnable)

# 1.1 Finitness is not a necessary condition for learnability

## 1.1.1 The set of threshold functions is PAC learnable

Consider the set of threshold functions over the real line <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}%3D\{h_a,a\in\mathbb{R}\}>,