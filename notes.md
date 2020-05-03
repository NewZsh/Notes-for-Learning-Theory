
- [1.1 loss function](#11-loss-function)
	- [1.1.1 softmax](#111-softmax)
  
# 1.1 loss function

## 1.1.1 softmax

https://zhuanlan.zhihu.com/p/45014864

https://kexue.fm/archives/3290

http://www.matrix67.com/blog/archives/2830

https://en.wikipedia.org/wiki/LogSumExp

Denote the number of classes to be $C$, a model outputs $C$ scores for each class, the learning objective is: **the target score must be larger than non-target scores**. Mathematically speaking, denote $z=f(x)\in\mathcal{R}^C$, and $y$ is the target class, then it must hold that

$$\forall j\neq y, \ z_y>z_y$$

The naive version of loss function is:

$$\sum_{i\neq y} \max(z_i-z_y, 0)$$

To improve the generalization, a margin is introduced:

$$\sum_{i\neq y} \max(z_i-z_y+m, 0)$$

However, this hinge loss leads to the optimizations of too many non-target scores. When the number of classes is too large, the model will suffer from gradient explosion.

To overcome this problem, we change a form of the training objective, that is: **the target score must be larger than the maximum of non-target scores**, then the loss function is:

$$\max(\max_{i\neq y} \{z_i\}-z_y, 0)$$

However, in each epoch, only two gradients of +1 and -1 make effect on backpropogation, and the learning procedure may be too slow. So we must smooth the maximum function. Note that, the smooth version of maximum funciton is LogSumExp (https://en.wikipedia.org/wiki/LogSumExp) but not softmax.

$$\max \left(\log \left( \sum_{i\neq y} e^{z_i} \right)-z_y, 0 \right)$$

also note that the derivative of LogSumExp() function is near to the softmax function.

To encourage $z_y$ (ensure generalization), continue smothing the ReLU $\max(x,0)$, that is, the softpluts function $\log(1+e^x)$.

$$\log \left( 1 + \exp \left[ \log \left( \sum_{i\neq y} e^{z_i} \right) - z_y\right] \right) = -\log \frac{e^{z_y}}{\sum_i e^{z_i}} $$

1.1.2 circle loss

adapt softmax in the case of multiple positive

$$ \mathcal{L}_u $$

$$ \mathcal{L} = \log \left[ 1+ \sum_{i=1}^K \sum_{j=1}^L \exp (\gamma (s_n^j-s_p^i+m))\right] = \log \left[ 1+ \sum_{j=1}^L \exp (\gamma (s_n^j+m))\sum_{i=1}^K \exp (\gamma (-s_p^i))\right] $$

$$ L_u = \log \left[ 1+ \sum_{i=1}^K \sum_{j=1}^L \exp (\gamma (s_n^j-s_p^i+m))\right] = \log \left[ 1+ \sum_{j=1}^L \exp (\gamma (s_n^j+m))\sum_{i=1}^K \exp (\gamma (-s_p^i))\right] $$

Is it balanced? ($K$ positive samples, $L$ negative samples)

$$L_u \sim \left[ \log \left\{ \sum_{j=1}^L \exp (\gamma (s_n^j+m))\sum_{i=1}^K \exp (\gamma (-s_p^i))  \right\}\right]_+ = \left[ \log \sum_{j=1}^L \exp (\gamma (s_n^j+m)) +  \log \sum_{i=1}^K \exp (\gamma (-s_p^i))  \right]_+ \sim \gamma  \left[\max s_n  - \min s_p + m \right]_+$$

