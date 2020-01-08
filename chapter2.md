**Notes of Statistical Machine Learning Theory**

*The notes is mainly based on the following book*

- UML: [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)  Shai Shalev-Shwartz and Shai Ben-David, 2014.
- PRML: [pattern recognition and machine learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) Christopher M. Bishop, 2006.
- PGM: [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models) Daphne Koller and Nir Friedman, 2009.
- GEV: [Graphical Models, Exponential Families, and Variational Inference](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf) Martin J. Wainwright and Michael I. Jordan, 2008.

**Chapter TWO VC-dimension**


- [1.1 The VC-dimension](#11-the-vc-dimension)
  - [1.1.1 Finitness is not a necessary condition for learnability](#111-finitness-is-not-a-necessary-condition-for-learnability)
  - [1.1.2 Shattering](#112-shattering)
  - [1.1.3 The VC-dimension](#113-the-vc-dimension)
  - [1.1.4 Examples](#114-examples)
- [1.2 Fundermental theorem of PAC learning](#12-fundermental-theorem-of-pac-learning)
  - [1.2.1 Equivalent statements](#121-equivalent-statements)
  - [1.2.2 Quantitative versions](#122-quantitative-versions)
- [1.3 Growth function](#13-growth-function)
  - [1.3.1 Growth function](#131-growth-function)
  - [1.3.2 Sauer's lemma](#132-sauers-lemma)
  - [1.3.3 Uniform convergence for classes of small effective size](#133-uniform-convergence-for-classes-of-small-effective-size)
- [1.4 Excercises and solutions](#14-excercises-and-solutions)

# 1.1 The VC-dimension

## 1.1.1 Finitness is not a necessary condition for learnability

Consider the set of threshold functions over the real line

<div align=center>
<img src=http://latex.codecogs.com/gif.latex?\mathcal{H}%3D\{h_a(x)%3D\mathbb{I}_{[x\leq%20a]},a\in\mathbb{R}\}>,
</div align=center>

Let <img src=http://latex.codecogs.com/gif.latex?a^*> be the threshold such that <img src=http://latex.codecogs.com/gif.latex?L_\mathcal{D}(h^*)%3D0>. Let <img src=http://latex.codecogs.com/gif.latex?a_0%3Ca^*%3Ca_1> such that

<div align=center>
<img src=http://latex.codecogs.com/gif.latex?\mathop{\mathbb{P}}\limits_{x\sim\mathcal{D}_x}[x\in(a_0,a^*)]%3D\mathop{\mathbb{P}}\limits_{x\sim\mathcal{D}_x}[x\in(a^*,a_1)]%3D\epsilon>,
</div align=center>

## 1.1.2 Shattering

The definition of VC-dimension is motivated from the No-Free-Lunch therorem: without restricting the hypothesis class, for any learning algorithm, an adversary can construct a distribution for which the learning algorithm will perform poorly, while there is another learning algorithm that will succeed on the same distribution.

## 1.1.3 The VC-dimension

## 1.1.4 Examples

# 1.2 Fundermental theorem of PAC learning

## 1.2.1 Equivalent statements

## 1.2.2 Quantitative versions

# 1.3 Growth function

## 1.3.1 Growth function

## 1.3.2 Sauer's lemma

## 1.3.3 Uniform convergence for classes of small effective size

        This part is not that important. The main conclusion is that 

# 1.4 Excercises and solutions