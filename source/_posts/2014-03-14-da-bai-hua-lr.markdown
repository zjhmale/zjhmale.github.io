---
layout: post
title: "大白话LR"
date: 2014-08-14 11:52:36 +0800
comments: true
categories: ml
---

### what is LR

逻辑斯蒂回归既可以解决分类问题也可以解决回归问题，正常情况下都是输出一个概率值，也就是这个值大于$$0.5$$那么认为输出是正例，如果小于$$0.5$$输出就是负例，正负例是和训练样本中的正负例对应的。一般都是解决二分类问题，也就是说一个预测值是$$1$$还是$$0$$。

逻辑斯蒂回归模型在做分类时，其实相当于是一个只有单个输入层单个输出层没有隐含层的感知器模型，多个输入节点通过权值将输入样本值传递给`Logistic`函数，然后得到最终的输出值，也就是一个在$$[0, 1]$$之间的概率值。

### how it works

`Logistic`函数的表达式或者也叫做`Sigmoid`函数

$$\frac{1}{1+e^{-z}}$$

上面说得多个输入节点通过权值将输入样本传递给输出层的节点的过程可以用下面的公式描述

$$\theta _{0} + \theta_{1}\cdot x_{1} + ... + \theta_{n} \cdot x_{n} = \sum_{i=0}^{n} \theta _{i} \cdot x_{i} = \theta ^{T} \cdot x$$

其中$$x=[x_{0}, x_{1} \; ... \; x_{n}]$$表示输入值向量，$$\theta=[\theta_{0}, \theta_{1} \; ... \; \theta_{n}]$$的是权值向量，要保证有一个常数项用来控制偏移，所以$$X_{0} = 1$$为，在上面的公式也体现了

然后通过将得到这个非激活值传递给`Logistic`函数得到最终的输出

$$h_{\theta} (x) = g(\theta _{0} + \theta_{1}\cdot x_{1} + ... + \theta_{n} \cdot x_{n}) = g(\theta ^{T} \cdot x) = \frac{1}{1+e^{-\theta ^{T} \cdot x}}$$

$$h_{\theta} (x)$$函数的物理意义在于其输出的值是当前输入样本数据是正例的概率，输出的值越接近于$$1$$，表示当前样本是正例的概率越大。

所以输入样本为正例的概率为

$$P(y=1|x;\theta) = h_{\theta}(x)$$

输入样本为负例的概率为

$$P(y=0|x;\theta) = 1-h_{\theta}(X)$$

两个式子可以合并为一个

$$P(y|x;\theta) = (h_{\theta}(x))^{y} \cdot (1-h_{\theta}(x))^{(1-y)}$$

确定了数据传递的关系之后，就可以构造损失函数开始训练了，根据`L2`范式，可以构造如下省略了正则化项的损失函数，其实可以从另一个角度理解LR的损失函数，其实就是让原本是负例的样本的输出是负例的概率尽可能大，原本是正例的样本经过输出之后是正例的概率也尽可能大，将每一个样本的概率乘起来的值要尽可能大才能保证模型可以拟合样本数据。

因为要使概率尽可能大，所以构造似然函数

$$L(\theta) = P(\vec{y}|X;\theta) = \prod_{i}^{m} P\left ( y^{(i)}|x^{(i)};\theta \right ) = \prod_{i}^{m} (h_{\theta}(x^{(i)}))^{y^{(i)}} \cdot (1-h_{\theta}(x^{(i)}))^{(1-y^{(i)})}$$

由于乘积项没有求和项好操作，所以我们可以取一个对数从而将乘积变为求和又不改变目标函数的单调性

$$l(\theta) = log(L(\theta)) = \sum_{i=1}^{m}y^{(i)} \cdot log(h(x^{(i)})) + (1-y^{(i)}) \cdot log(1-h(x^{(i)}))$$

如果按照这个损失函数去训练的话，我们需要找到这个函数的全局最大值或者是某一个局部最大值，目前没有行之有效的算法，但是如果把问题转换成一个求最小值的问题，就可以使用梯度下降算法来进行训练了

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m}Cost(h_{\theta}(x^{(i)}), y^{(i)}) = -\frac{1}{m}[\sum_{i=1}^{m}y^{(i)} \cdot log(h(x^{(i)})) + (1-y^{(i)}) \cdot log(1-h(x^{(i)}))]$$

至此，已经构造出了用于训练的损失函数，下面就可以使用梯度下降来推导权值更新的公式了，我们定义$$\alpha$$为梯度下降的学习速率

$$\begin{align*}
\frac{\partial J(\theta) }{\partial \theta_{j}} &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{h_{\theta}(x_{(i)})}\frac{\partial h_{\theta}(x^{(i)})}{\partial \theta_{j}} - (1-y^{(i)})\frac{1}{1-h_{\theta}(x^{(i)})}\frac{\partial h_{\theta}(x^{(i)})}{\partial \theta_{j}}) \\ &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{g(\theta^{T}x^{(i)})}-(1-y^{(i)})\frac{1}{1-g(\theta^{T}x^{(i)})})\frac{\partial g(\theta^{T}x^{(i)})}{\partial \theta_{j}} \\ &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{g(\theta^{T}x^{(i)})}-(1-y^{(i)})\frac{1}{1-g(\theta^{T}x^{(i)})})g(\theta^{T}x^{(i)})(1-g(\theta^{T}x^{(i)}))\frac{\partial \theta^{T}x^{(i)}}{\partial \theta_{j}} \\ &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}(1-g(\theta^{T}x^{(i)})) - (1-y^{(i)})g(\theta^{T}x^{(i)})))x_{j}^{(i)} = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-g(\theta^{T}x^{(i)}))x_{j}^{(i)} \\ &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)} \\ &= \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}
\end{align*}$$

上面的推导中用到了`Logistic`函数求导的性质

上面公式已经推导出了梯度值，而梯度下降就是沿着负梯度的方向优化，所以最终权值更新的公式就是

$$\theta_{j} := \theta_{j} - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}, \; (j = 0 \; ... \; n)$$

因为$$\frac{1}{m}$$和$$\alpha$$都是常数，所以这两项可以合并为一个常数项$$\alpha$$，更新公式可以最终化简为

$$\theta_{j} := \theta_{j} - \alpha\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}, \; (j = 0 \; ... \; n)$$

之后就可以进行批量梯度下降或者是随机梯度下降了

### how to implement vectorization

在批量梯度下降算法中，如果要进行一次迭代更新需要进行遍历操作，需要许多额外的代码，由于LR中的所有项都可以表示为向量，所以也可以推导出一个向量化或者说是矩阵化的学习算法实现

首先定义训练样本集，$$x$$表示样本集中的输入值，$$y$$代表输出，可以看出有$$m$$个样本，每一个样本中有$$n$$个特征

$$x = \begin{bmatrix}
x^{(1)}\\ 
x^{(2)}\\ 
...\\ 
x^{(m)}
\end{bmatrix}
=
\begin{bmatrix}
x_{0}^{(1)} & x_{1}^{(1)} & ... & x_{n}^{(1)}\\ 
x_{0}^{(2)} & x_{1}^{(2)} & ... & x_{n}^{(2)}\\ 
... & ... & ... & ...\\ 
x_{0}^{(m)} & x_{1}^{(m)} & ... & x_{n}^{(m)}
\end{bmatrix}
, \;
y=\begin{bmatrix}
y^{(1)}\\ 
y^{(2)}\\ 
...\\ 
y^{(m)}
\end{bmatrix}$$

一次权值向量也可以表示为

$$\theta = \begin{bmatrix}
\theta_{1}\\ 
\theta_{2}\\ 
...\\ 
\theta_{n}
\end{bmatrix}$$

将输入给`Logistic`的为激活值表示为矩阵形式$$A$$

$$
A = x\cdot \theta = \begin{bmatrix}
x_{0}^{(1)} & x_{1}^{(1)} & ... & x_{n}^{(1)}\\ 
x_{0}^{(2)} & x_{1}^{(2)} & ... & x_{n}^{(2)}\\ 
... & ... & ... & ...\\ 
x_{0}^{(m)} & x_{1}^{(m)} & ... & x_{n}^{(m)}
\end{bmatrix} \cdot 
\begin{bmatrix}
\theta_{1}\\ 
\theta_{2}\\ 
...\\ 
\theta_{n}
\end{bmatrix}
=
\begin{bmatrix}
\theta_{0}x_{0}^{(1)} & \theta_{0}x_{1}^{(1)} & ... & \theta_{n}x_{n}^{(1)}\\ 
\theta_{0}x_{0}^{(2)} & \theta_{0}x_{1}^{(2)} & ... & \theta_{n}x_{n}^{(2)}\\ 
... & ... & ... & ...\\ 
\theta_{0}x_{0}^{(m)} & \theta_{0}x_{1}^{(m)} & ... & \theta_{n}x_{n}^{(m)}
\end{bmatrix}
$$

计算误差为$$E$$

$$
E = h_{\theta}(x) - y =g(A)-y = \begin{bmatrix}
g(A^{(1)}) - y^{(1)}\\ 
g(A^{(2)}) - y^{(2)}\\ 
...\\ 
g(A^{(m)})-y^{(m)}
\end{bmatrix}
=
\begin{bmatrix}
e^{(1)}\\ 
e^{(2)}\\ 
...\\ 
e^{(m)}
\end{bmatrix}
$$

所以可以根据上面更新单个权值的公式很容易地推导出矩阵形式的更新公式

$$
\begin{bmatrix}
\theta_{1}\\ 
\theta_{2}\\ 
...\\ 
\theta_{n}
\end{bmatrix}
:=
\begin{bmatrix}
\theta_{1}\\ 
\theta_{2}\\ 
...\\ 
\theta_{n}
\end{bmatrix}
-
\alpha \cdot \begin{bmatrix}
x_{0}^{(1)} & x_{0}^{(2)} & ... & x_{0}^{(m)}\\ 
x_{1}^{(1)} & x_{1}^{(2)} & ... & x_{1}^{(m)}\\ 
... & ... & ... & ...\\ 
x_{n}^{(1)} & x_{n}^{(2)} & ... & x_{n}^{(m)}
\end{bmatrix}
\cdot
E
$$

最终向量化的更新公式化简后如下所示

$$\theta := \theta - \alpha \cdot (\frac{1}{m}) \cdot x^{T} \cdot (g(x\cdot \theta) - y)$$

所以向量化的更新步骤可以描述如下(_上文也提到了1/m是可以省略并入一个常数项的_)

1. 求$$A = x \cdot \theta$$
2. 求$$E = g(A) - y$$
3. 求$$\theta := \theta - \alpha \cdot x^{T} \cdot E$$

<!--$$\begin{align*}
P(\exists h \in \mathcal{H}. \vert \epsilon(h_i) - \hat{\epsilon(h_i)} \vert > \gamma) & = P (A_1 \cup \cdots \cup A_k) \\
 & \le \sum_{i=1}^k P(A_i) \\
 & \le \sum_{i=1}^k 2 \exp(-2 \gamma^2 m) \\
 & = 2k \exp(-2 \gamma^2 m)
\end{align*}$$


$$\begin{align*}
\frac{\partial J(\theta) }{\partial \theta_{j}} &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{h_{\theta}(x_{(i)})}\frac{\partial h_{\theta}(x^{(i)})}{\partial \theta_{j}} - (1-y^{(i)})\frac{1}{1-h_{\theta}(x^{(i)})}\frac{\partial h_{\theta}(x^{(i)})}{\partial \theta_{j}}) \\ &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{g(\theta^{T}x^{(i)})}-(1-y^{(i)})\frac{1}{1-g(\theta^{T}x^{(i)})})\frac{\partial g(\theta^{T}x^{(i)})}{\partial \theta_{j}} \\ &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{g(\theta^{T}x^{(i)})}-(1-y^{(i)})\frac{1}{1-g(\theta^{T}x^{(i)})})g(\theta^{T}x^{(i)})(1-g(\theta^{T}x^{(i)}))\frac{\partial \theta^{T}x^{(i)}}{\partial \theta_{j}} \\ &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}(1-g(\theta^{T}x^{(i)})) - (1-y^{(i)})g(\theta^{T}x^{(i)})))x_{j}^{(i)} = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-g(\theta^{T}x^{(i)}))x_{j}^{(i)} \\ &= -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)} \\ &= \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}
\end{align*}$$-->
