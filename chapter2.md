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

If <img src=http://latex.codecogs.com/gif.latex?\mathcal{D}_x(-\infty,a^*)\leq\epsilon> we set <img src=http://latex.codecogs.com/gif.latex?a_0%3D-\infty>, and similarily for <img src=http://latex.codecogs.com/gif.latex?a_1>.

Given a training set <img src=http://latex.codecogs.com/gif.latex?S>, let <img src=http://latex.codecogs.com/gif.latex?b_0%3D\max\{x:(x,1)\in%20S\}> (if no example is positive then <img src=http://latex.codecogs.com/gif.latex?b_0%3D-\infty>), and <img src=http://latex.codecogs.com/gif.latex?b_1%3D\min\{x:(x,0)\in%20S\}> (if no example is negative then <img src=http://latex.codecogs.com/gif.latex?b_1%3D\infty>). Let <img src=http://latex.codecogs.com/gif.latex?b_S> be the threshold of an ERM hypothesis <img src=http://latex.codecogs.com/gif.latex?h_S>, which implies <img src=http://latex.codecogs.com/gif.latex?b_S\in(b_0,b_1)>, then we have

<div align=center>
<img src=http://latex.codecogs.com/gif.latex?\mathop{\mathbb{P}}\limits_{S\sim\mathcal{D}^m}[L_\mathcal{D}(h_S)%3C\epsilon]\leq\mathop{\mathbb{P}}\limits_{S\sim\mathcal{D}^m}[b_0%3Ca_0]+\mathop{\mathbb{P}}\limits_{S\sim\mathcal{D}^m}[b_1%3Ea_1]>,
</div align=center>

Each term on the right-side is bounded by <img src=http://latex.codecogs.com/gif.latex?(1-\epsilon)^m\leq%20e^{-\epsilon%20m}>. Let <img src=http://latex.codecogs.com/gif.latex?m%3E\log(2/\delta)/\epsilon>, then the left-side is bounded by <img src=http://latex.codecogs.com/gif.latex?\delta>. As a result, the hypothesis class is PAC-learnable.

## 1.1.2 Shattering

The definition of VC-dimension is motivated from the No-Free-Lunch therorem: without restricting the hypothesis class, for any learning algorithm, an adversary can construct a distribution for which the learning algorithm will perform poorly, while there is another learning algorithm that will succeed on the same distribution. To make any algorithm fail, the adversary used the power of choosing a target function from the set of all possible labelling functions.

When considering PAC learnability of a hypothesis class <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}>, the adversary is restricted to constructing distributions for which some hypothesis <img src=http://latex.codecogs.com/gif.latex?h\in\mathcal{H}> achieves a zero risk. Since we are considering distributions that are concentrated on elements of <img src=http://latex.codecogs.com/gif.latex?C>, we should study how <img src=http://latex.codecogs.com/gif.latex?h\in\mathcal{H}> behaves on <img src=http://latex.codecogs.com/gif.latex?C>.

**definition: Restriction of <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}> to <img src=http://latex.codecogs.com/gif.latex?C>**. The restriction of <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}> to <img src=http://latex.codecogs.com/gif.latex?C> is the set of functions from <img src=http://latex.codecogs.com/gif.latex?C> to <img src=http://latex.codecogs.com/gif.latex?\{0,1\}> that can be derived from <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}>. That is,

<div align=center>
<img src=http://latex.codecogs.com/gif.latex?\mathcal{H}_C%3D\{(h(c_1),\cdots,h(c_m):h\in\mathcal{H}\}>,
</div align=center>

where we represent each function from <img src=http://latex.codecogs.com/gif.latex?C> to <img src=http://latex.codecogs.com/gif.latex?\{0,1\}> as a vector in <img src=http://latex.codecogs.com/gif.latex?\{0,1\}^{|C|}>.

**definition: Shattering**. A hypothesis class <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}> shatters a finite set <img src=http://latex.codecogs.com/gif.latex?C\in\mathcal{X}> if the restriction of <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}> to <img src=http://latex.codecogs.com/gif.latex?C> is the set of all functions from <img src=http://latex.codecogs.com/gif.latex?C> to <img src=http://latex.codecogs.com/gif.latex?\{0,1\}>. That is, <img src=http://latex.codecogs.com/gif.latex?|\mathcal{H}_C|%3D2^{|C|}>.

## 1.1.3 The VC-dimension

**definition: VC-dimension**. The VC-dimension of a hypothesis class <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}>, denoted <img src=http://latex.codecogs.com/gif.latex?\text{VCdim}(\mathcal{H})>, is the maximal size of a set <img src=http://latex.codecogs.com/gif.latex?C\subset\mathcal{X}> that can be shattered by <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}>. If <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}> can shatter sets of arbitrarily large size we say that <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}> has infinite VC-dimension.

To calculate the VC-dimension for a hypothesis set, we should show that

  - There **exists** a subset of size <img src=http://latex.codecogs.com/gif.latex?d> that can be shattered
  - **Every** subset of size <img src=http://latex.codecogs.com/gif.latex?d+1> can not be shattered

1. Example1: threshold functions
  
    For an arbitary set <img src=http://latex.codecogs.com/gif.latex?\{c\}>, it can be shattered by <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}>, therefore <img src=http://latex.codecogs.com/gif.latex?\text{VCdim}(\mathcal{H})\geq%201>;

    For an arbitary set <img src=http://latex.codecogs.com/gif.latex?\{c_1,c_2\}>, where <img src=http://latex.codecogs.com/gif.latex?c_1\leq%20c_2>, any threshold that assigns 0 to <img src=http://latex.codecogs.com/gif.latex?c_1> must assign 0 to <img src=http://latex.codecogs.com/gif.latex?c_2>, so not all functions from <img src=http://latex.codecogs.com/gif.latex?\mathcal{C}> to <img src=http://latex.codecogs.com/gif.latex?\{0,1\}> are included by <img src=http://latex.codecogs.com/gif.latex?\mathcal{H}_C>. Therefore it can not be shattered.

    Hence in conclusion, the VC-dimention of the class of threshold functions is 1.

2. Example2: Intevals

    For 

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