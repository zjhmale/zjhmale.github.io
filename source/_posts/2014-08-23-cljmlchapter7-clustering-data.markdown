---
layout: post
title: "cljmlchapter7-clustering data"
date: 2014-08-23 09:44:20 +0800
comments: true
categories: clojure fp
---

在这一章中我们将把我们的注意力集中在**非监督学习(unsupervised learning)**问题上。我们会学习几个**聚类(clustering)**算法，或者说是**聚类器(clusterers)**的原理，并使用Clojure实现它们。此外还会展示几个已经完整实现了聚类算法的Clojure库。直到本章的最后，我们将会探究**降维(dimensionality reduction)**技术，以及如何使用这种技术来对给定的高维样本数据提供一个易于理解的可视化效果。

聚类或者说**聚类技术(cluster analysis)**是一种将数据或者样本进行分组的基本方法。作为一种非监督学习的形式，一个聚类模型使用无标记的数据来进行训练，意味着训练数据中的样本不会带有输入特征值序列对应的类别，训练数据中对于给定的一组输入值并没有描述对应的输出值。一个聚类模型必须确定多组输入特征值之间的相似性，然后自动推断出每一个样本最可能属于的类别。因此使用这一类模型可以将样本数据分成多个群体(簇)。

在解决现实世界中的问题时聚类技术也有若干实际应用。聚类通常被用于图像分析，图像分割，软件演化系统以及社交网络分析。此外聚类不仅广泛应用在计算机科学领域，在生物学分析，基因分析以及犯罪分析等领域也有广泛的应用。

到目前为止已经有好几种聚类算法已经公开并广泛应用在学术界与工业界。而每一种算法都有一组独立的概念，来定义一个簇以及如何将不同的样本组合成一个新的簇。但是不幸的是，目前并没有一种直接的指导方案来描述对于给定的数据集应该用哪一种聚类算法去建模，需要将选定的几个聚类算法作用在数据集上进行试验并计算误差来进行评估，以确定使用一种最合适的聚类算法来聚类给定的样本数据。

这是因为在聚类问题中输入的数据是无标记的，也就是说在输入数据中并没有标出样本对应的类别，也无法使用简单的基于是/否奖励机制的训练来推断出样本属于的类别。

在这一章中，我们将会了解少数的几个可以用在非标记数据集上的聚类技术。

## 使用K-means聚类

**K-means聚类**算法是一种基于矢量量化的聚类技术(更多信息可以参考"Algorithm AS 136: A K-Means Clustering Algorithm"这篇论文)。这个算法将一系列的以向量形式表示的样本分成K个样本簇，所以也因此而得名。在这一章中我们将学习K-means算法的性质，并且尝试将其实现。

**矢量量化(Vector Quantization)**技术通常用于信号处理领域，用来将一个包含很多值的大集合中的值映射到一个只含有少量值的空间中的值。例如，一个模拟信号用8位的数字信号来量化，那么可以这个模拟信号中可能有256个不同的量化等级，也就是有256个值。而假如需要将原本$$[0, 255]$$上的值映射到$$[0, 5]$$的范围内，一个简单的做法就是将原本的信号值乘上$$\frac{5}{256}$$，从而将信号值映射到新的范围内。在聚类的情境中，对输入数据或者输出数据进行矢量量化可行的原因如下：

* 可以将聚类行为限制到一个有限的集簇内。
* 在聚类过程中需要适应在一定范围内的样本数据并不会每次都聚集到同一个簇中，而这种灵活性决定了聚类算法可以对没有对应类别的样本值进行分组。

这个算法的要点可以简单地概括如下。这个K个均值点，或者说重心点，一开始是被随机选择出来的。然后计算除了选出的重心点以外的所有样本点到这几个重心点的距离。然后一个样本点将被分组到离这个样本点最近的重心点代表的簇中。在一个多维空间中，每一个样本数据都有多个维度的特征值，将使用**欧式距离(Euclidean distance)**来计算各个输入向量到给定的重心点，这一步在K-means算法中称作**分配步骤**。

K-means算法中的下一步称为**更新步骤**。此时K个重心点的坐标位置不再是一开始随机选出的坐标值，而是利用之前分组之后的每一簇中的所有样本点坐标值平均值来作为新的重心点的坐标值。然后重复迭代这两部操作，直到在相邻的两次更新中重心坐标的偏移量基本可以忽略不计。因此这个算法的最终结果是给定的训练数据中每一组输入特征值都被分到一个簇，或者说一个类别中。

K-means算法的迭代过程可以用下面的一组图来可视化地展示：

<center>
	<img src="/images/cljml/chap7/image1.png">
</center>

上面组图中的每一张图都表示了K-means算法对于给定训练数据在每一次迭代后重心的坐标位置以及其余样本的分组情况。最后一张图表示了最后一次迭代之后K-means算法聚类出的不同簇的位置。

K-means聚类算法的优化目标可以用如下表达式形式化地描述：

$$\underset{s}{arg \; min} \sum_{i=1}^{K} \sum_{X_{j} \epsilon S_{i}} \left \| X_{j} - \mu _{i} \right \|^{2}$$

在上面表达式中定义的最优化问题中，$$\mu_{1}, \mu_{2}, \cdots, \mu_{k}$$这几项代表了用来聚类输入值的簇的重心坐标值。K-means算法可以最小化每一个簇的大小，同时也确定了每一个簇的重心位置。

这个算法需要$$N$$个样本数据以及$$K$$个初始的随机均值点作为算法的输入。在最开始的分配步骤中，输入样本值被分配到以传给算法的初始随机均值点为重心位置的簇中。在之后的更新步骤中，利用初始聚类的样本数据来计算每一个簇的新的重心位置。在大部分的K-means算法的实现中，都是利用某一个簇中所有样本点坐标的均值来作为这个簇新的重心位置。

大多数的K-means算法实现中都是从训练数据集中随机的选择$$K$$个样本点最为初识情况下的$$K$$个簇的重心点。这种技术也称作**Forgy**随机初始化方法。

假如要聚类的簇的个数$$K$$是无限大的或者输入样本数据中的特征维度$$d$$是无限大的，那么K-means算法是一个NP-hard问题。当这两个值都确定了边界范围之后，K-means算法的时间复杂度是$$O(n^{d(k+1)}log^{n})$$。根据计算新的均值的方法不同，K-means算法还有几个不同的变种。

现在将会演示如何不使用额外的依赖库，只是用简单的Clojure代码来实现K-means算法。我们会先定义这个算法的所有细节和小模块，然后将这些小模块组合在一起从而得到一个基本的K-means算法。

两个数字之间的距离可以被定义为这两个数字的值差值的绝对值，我们可以实现一个`distance`函数来计算两个数的距离，如下面代码所示：

{% codeblock lang:clojure %}
(defn distance [a b]
  (if (< a b) (- b a) (- a b)))
{% endcodeblock %}

假如我们给定一组平均值，我们可以计算出哪一个平均值离给定的数字最近，我们可以利用`distance`函数和`sort-by`函数来做到这一点，如下面代码所示：

{% codeblock lang:clojure %}
(defn closest [point means distance]
  (first (sort-by #(distance % point) means)))
{% endcodeblock %}

为了演示如何使用上面代码定义的`closest`函数，我首先需要定义一些样本数据放在一个序列中表示，然后再定义一些均值也同样放在序列中表示，如下面代码所示：

{% codeblock lang:clojure %}
(def data '(2 3 5 6 10 11 100 101 102))
(def guessed-means '(0 10))
{% endcodeblock %}

现在我们可以在REPL中将任意数字和`guessed-means`变量传入`closest`函数，从而检查`closest`函数的行为：

{% codeblock lang:clojure %}
user> (closest 2 guessed-means distance)
0
user> (closest 9 guessed-means distance)
10
user> (closest 100 guessed-means distance)
10
{% endcodeblock %}

我们给定的均值是0和10，所以将2传入后`closest`函数会返回距离2最近的均值0，而将9和100传入则会返回距离这两个给定数值较近的均值10。因此，给定的数据集中的样本数据可以被分组到距离最近的均值对应的簇中。我们可以使用`closest`函数和`group-by`函数来实现一个函数来执行分组的行为，如下面代码所示：

{% codeblock lang:clojure %}
(defn point-groups [means data distance]
  (group-by #(closest % means distance) data))
{% endcodeblock %}

上面代码定义的`point-groups`函数需要传入三个参数，第一个参数是给定的一组均值，第二个参数是待分组的数据集，最后一个参数是用来计算待分组数据集中样本数据与均值之间距离的函数行为。需要注意的是，在使用`group-by`函数时需要两个参数，第一个参数是一个函数闭包，第二个参数是一个数据序列，通过将数据序列中的值不断地传入函数闭包中得到计算结果，最终利用得到的计算结果来对数据序列分组。

我们可以将`data`变量表示的待分组数据集，`guessed-means`变量表示的均值序列还有用来计算数值距离的`distance`函数传入`point-groups`函数，从而对给定的待分组数据集进行简单的聚类，如下面代码所示：

{% codeblock lang:clojure %}
user> (point-groups guessed-means data distance)
{0 [2 3 5], 10 [6 10 11 100 101 102]}
{% endcodeblock %}

如上面代码所示，`point-groups`函数将用`data`变量表示的待分组数据集划分到了两个组别中。为了计算分出来的两个组别中新的均值，我们可以分别计算每一个组中所有数值的平均值，可以使用`reduce`函数和`count`函数来实现这一步操作，如下面代码所示：

{% codeblock lang:clojure %}
(defn average [& list]
  (/ (reduce + list)
     (count list)))
{% endcodeblock %}

我们会实现一个函数来将上面代码定义的`average`函数作用在之前的均值序列以及`point- groups`函数返回的用map对象表示的分组结果上，从而计算每一个分组中新的均值：

{% codeblock lang:clojure %}
(defn new-means [average point-groups old-means]
  (for [m old-means]
    (if (contains? point-groups m)
      (apply average (get point-groups m))
      m)))
{% endcodeblock %}

上面代码中定义的`new-means`函数中，对于每一个之前均值序列中的均值，我们将`average`函数作用在这个均值对应的簇中的所有样本点组成的序列上。当然在使用`average`函数之前，我们需要先判断当前选择出来的均值是否出现在之前的分组结果中，也就是这个均值对应的簇中是否有样本数据点，我们`new-means`函数中使用`contains?`函数来做这个验证检查。我们可以在REPL中查看`new-means`函数作用在上一个均值序列以及上一次分组结果后返回的结果值，如下所示：

{% codeblock lang:clojure %}
user> (new-means average
           (point-groups guessed-means data distance)
           guessed-means)
(10/3 55)
{% endcodeblock %}

如上面例子的输出所示，我们利用`new-means`函数将一开始的均值序列`(0 10)`变为了新的均值序列`(10/3 55)`。为了实现K-means算法，我们需要把`new-means`函数不断地作用在上一次迭代得到的均值序列上。这个迭代过程可以使用`iterate`函数来实现，其中`iterate`函数需要传入一个单参数的函数闭包。

我们可以先定义要传给`iterate`函数的闭包，在这个闭包中，我们将上一次迭代的均值序列传给`new-means`函数，从而得到新的均值序列。如下面代码所示：

{% codeblock lang:clojure %}
(defn iterate-means [data distance average]
  (fn [means]
    (new-means average
               (point-groups means data distance) means)))
{% endcodeblock %}

上面代码中定义的`iterate-means`会返回一个新的匿名函数，这个匿名函数接受上一次的均值序列作为参数，然后利用这个匿名函数可以计算出新的均值序列，我们可以在REPL中查看这个返回的结果：

{% codeblock lang:clojure %}
user> ((iterate-means data distance average) '(0 10))
(10/3 55)
user> ((iterate-means data distance average) '(10/3 55))
(37/6 101)
{% endcodeblock %}

从上面的输出结果可以看到，每一次使用`iterate-means`函数，都可以得到一个新的均值序列，其中`iterate-means`函数返回匿名函数就可以当做函数闭包传入`iterate`函数，而`iterate`函数会返回一个无穷大的包含每一次得到均值序列的序列，所以我们需要使用`take`函数来查看`iterate`函数返回的结果，从而查看每一次迭代得到的均值序列，如下面代码所示：

{% codeblock lang:clojure %}
user> (take 4 (iterate (iterate-means data distance average)
                        '(0 10)))
((0 10) (10/3 55) (37/6 101) (37/6 101))
{% endcodeblock %}

从上面的结果中可以观察到，对于我们给定的训练样本数据，均值序列仅仅在前三次迭代中发生了改变，最终收敛到了`(37/6 10)`的位置。K-means算法的终止条件是均值序列可以收敛到一个相对稳定的值，因此我们必须不断地将由`iterate-means`函数生成的函数闭包新生成的均值序列传入函数闭包中做下一次迭代，直到连续的两次迭代过程中得到的均值序列不再改变或者变化的偏差在一个可容忍的范围内。因为`iterate`函数会惰性地返回一个无穷大的序列，所以我们必须实现一个函数来根据均值序列收敛的位置来限制`iterate`返回的无穷大惰性序列的长度。我们可以使用`lazy-seq`函数和`seq`函数以惰性唤醒的方式来实现这种行为，如下面代码所示：

{% codeblock lang:clojure %}
(defn take-while-unstable
  ([sq] (lazy-seq (if-let [sq (seq sq)]
                    (cons (first sq)
                          (take-while-unstable
                           (rest sq) (first sq))))))
  ([sq last] (lazy-seq (if-let [sq (seq sq)]
                         (if (= (first sq) last)
                           nil
                           (take-while-unstable sq))))))
{% endcodeblock %}

上面代码定义的`take-while-unstable`函数将一个惰性序列分割成头部和尾部两部分，然后比较头部元素和尾部第一个元素是否相同，假如两个元素相同，则返回一个空序列或者`nil`，如果两个元素不相同，就将惰性序列的尾部作为参数递归调用`take-while-unstable`函数。使用`lazy-seq`的目的是为了将返回的序列变为惰性的从而可以节省不必要的求值计算，注意到我们还使用了`if-let`这个宏，这个宏其实就是一个简单的`let`形式加上一个`if`表达式，因为假如参数`sq`是一个空序列的话，那么`(seq sq)`会返回`nil`，所以如果使用`if-let`宏的话可以更优雅简洁的判断是否到了递归的种植条件。我们可以在REPL中查看`take-while-unstable`函数的行为以及这个函数的输出：

{% codeblock lang:clojure %}
user> (take-while-unstable
       '(1 2 3 4 5 6 7 7 7 7))
(1 2 3 4 5 6 7)
user> (take-while-unstable
       (iterate (iterate-means data distance average)
                '(0 10)))
((0 10) (10/3 55) (37/6 101))
{% endcodeblock %}

使用最终计算得到的均值序列，我们可以使用`vals`函数从`point-groups`函数返回的以map对象形式存储的聚类结果中得到每一个簇中所含有的数据样本，如下面代码所示：

{% codeblock lang:clojure %}
(defn k-cluster [data distance means]
  (vals (point-groups means data distance)))
{% endcodeblock %}

需要注意的是`vals`以序列形式返回一个map对象中所有的值。

上面代码定义的`k-cluster`函数利用K-means算法得到给定输入数据集的最终聚类结果。我们可以将最终迭代得到的均值序列`(37/6 101)`传入`k-cluster`函数来得到给定输入数据集的最终聚类结果，如下面代码所示：

{% codeblock lang:clojure %}
user> (k-cluster data distance '(37/6 101))
([2 3 5 6 10 11] [100 101 102])
{% endcodeblock %}

为了可以将给定数据集中的样本在聚类过程中聚类结果的变化直观地显示出来，我们可以将`k-cluster`函数作用在之前定义的由`iterate`函数和`iterate-means`函数组成的迭代过程返回的含有每一次迭代的均值序列的结果序列上。此外我们还要利用`take-while-unstable`函数来判断序列的收敛点从而限制返回序列的长度。整个过程如下面代码所示：

{% codeblock lang:clojure %}
user> (take-while-unstable
          (map #(k-cluster data distance %)
               (iterate (iterate-means data distance average)
                '(0 10))))
(([2 3 5] [6 10 11 100 101 102])
([2 3 5 6 10 11] [100 101 102]))
{% endcodeblock %}

我们可以将上述代码所示的表达式进行重构为一个函数，这个函数(准确的说是一个闭包)只需要一个参数来表示初识时的均值序列，函数内部中将这个均值序列和待聚类数据集以及用来计算数值之间距离的函数还有计算一个序列中所有值的均值的函数传给`iterate-means`来进行迭代计算。如下面代码所示：

{% codeblock lang:clojure %}
(defn k-groups [data distance average]
  (fn [guesses]
    (take-while-unstable
     (map #(k-cluster data distance %)
          (iterate (iterate-means data distance average)
                   guesses)))))
{% endcodeblock %}

我们可以将一开始定义的待聚类数据集以及两个作用在数值变量上的函数`distance`和`average`传入上面代码定义的`k-groups`函数，如下面代码所示：

{% codeblock lang:clojure %}
(def grouper
  (k-groups data distance average))
{% endcodeblock %}

现在我们可以把任意的初始均值序列作为参数传给上面代码定义的`grouper`函数，从而可以看到对不同的初始均值序列，K-means算法在聚类的过程中迭代的次数不同，最终收敛到的均值序列的值也不同，也就是说最后的聚类结果也不同，我们可以在REPL中观察不同初始均值序列下K-means算法聚类的迭代过程，每一次迭代的聚类结果以及最终的聚类结果：

{% codeblock lang:clojure %}
user> (grouper '(0 10))
(([2 3 5] [6 10 11 100 101 102])
 ([2 3 5 6 10 11] [100 101 102]))
user> (grouper '(1 2 3))
(([2] [3 5 6 10 11 100 101 102])
 ([2 3 5 6 10 11] [100 101 102])
 ([2 3] [5 6 10 11] [100 101 102])
 ([2 3 5] [6 10 11] [100 101 102])
 ([2 3 5 6] [10 11] [100 101 102]))
user> (grouper '(0 1 2 3 4))
(([2] [3] [5 6 10 11 100 101 102])
 ([2] [3 5 6 10 11] [100 101 102])
 ([2 3] [5 6 10 11] [100 101 102])
 ([2 3 5] [6 10 11] [100 101 102])
 ([2] [3 5 6] [10 11] [100 101 102])
 ([2 3] [5 6] [10 11] [100 101 102]))
{% endcodeblock %}

正如之前提到的，假如给定的初始均值序列中均值的个数大于待聚类数据集中样本的个数，那么我们最终得到的聚类结果就是，最终聚类得到的簇的个数与待聚类数据集中样本的个数相同，也就是说每一个簇中只有一个样本。我们可以在REPL中用`grouper`函数验证这一点，如下面代码所示：

{% codeblock lang:clojure %}
user> (grouper (range 200))
(([2] [3] [100] [5] [101] [6] [102] [10] [11]))
{% endcodeblock %}

我们可以将之前实现的算法扩展到向量值上，而不只是单单作用于简单的数值上，为了做到这一点我们需要修改作为参数传入`k-groups`函数的`distance`函数和`average`函数。这两个函数扩展到向量空间下的实现如下面代码所示：

{% codeblock lang:clojure %}
(defn vec-distance [a b]
  (reduce + (map #(* % %) (map - a b))))

(defn vec-average [& list]
  (map #(/ % (count list)) (apply map + list)))
{% endcodeblock %}

上面代码所定义的`vec-distance`函数利用计算两个向量对应位置元素的差值并且求取这些差值的平方和的方法实现了求取两个向量的平方欧氏距离。上面代码定义的`vec-average`函数是将多个向量相加，然后将相加得到的向量中的每一个元素都除以向量的个数的方法来实现求取多个向量的平均值。我们可以在REPL中观察这两个函数的返回值，如下所示：

{% codeblock lang:clojure %}
user> (vec-distance [1 2 3] [5 6 7])
48
user> (vec-average  [1 2 3] [5 6 7])
(3 4 5)
{% endcodeblock %}

现在，可以为我们的聚类算法定义一些向量空间下的值来作为待分类的样本数据集，如下所示：

{% codeblock lang:clojure %}
(def vector-data
  '([1 2 3] [3 2 1] [100 200 300] [300 200 100] [50 50 50]))
{% endcodeblock %}

现在我们可以将`vector-data`变量，以及`vec-distance`和`vec-average`两个函数传给`k-groups`函数，从而可以看到在待聚类样本是在向量空间下时聚类过程中每一次迭代的聚类结果以及最终算法收敛时的聚类结果，如下所示：

{% codeblock lang:clojure %}
user> ((k-groups vector-data vec-distance vec-average)
       '([1 1 1] [2 2 2] [3 3 3]))
(([[1 2 3] [3 2 1]] [[100 200 300] [300 200 100] [50 50 50]])
 ([[1 2 3] [3 2 1] [50 50 50]]
  [[100 200 300] [300 200 100]])
 ([[1 2 3] [3 2 1]]
  [[100 200 300] [300 200 100]]
  [[50 50 50]]))
{% endcodeblock %}

对于当前算法的实现中另一个可以改进的点是假如传入给`new-means`函数的均值序列中含有相同的值我们需要`new-means`函数每次只更新均值序列中具有相同值的元素中的一个元素的值。现在假如我们向`new-means`函数中传入一个均值序列，那么如果这个序列中有相同值的元素，这些具有相同值的元素都会被更新。然而，在经典的K-means算法中，每一次更新仅允许更新均值序列中相同值元素中的其中一个值。我们可以在REPL中验证当前算法存在的缺陷，我们向`new-means`函数传递`'(0 0)`作为初始的均值序列，如下面代码所示：

{% codeblock lang:clojure %}
user> (new-means average
                 (point-groups '(0 0) '(0 1 2 3 4) distance)
                 '(0 0))
(2 2)
{% endcodeblock %}

我们可以通过检查给定的均值在均值序列中是否重复出现，如果重复出现那么只更新这些具有相同值的均值中的其中一个来避免这个问题。我们可以利用`frequencies`函数来实现这个功能。`frequencies`函数接受一个序列，然后统计这个序列中每一个元素重复出现的次数，最后返回一个map对象，这个map对象中的键即为序列中的值，而建对应的值即为序列中每一个值重复出现的次数。因此，我们可以重新定义`new-means`函数，如下面代码所示：

{% codeblock lang:clojure %}
(defn update-seq [sq f]
  (let [freqs (frequencies sq)]
    (apply concat
           (for [[k v] freqs]
             (if (= v 1)
               (list (f k))
               (cons (f k) (repeat (dec v) k)))))))

(defn new-means [average point-groups old-means]
  (update-seq
   old-means
   (fn [o]
     (if (contains? point-groups o)
       (apply average (get point-groups o)) o))))
{% endcodeblock %}

上面代码定义的`update-seq`函数接受一个函数闭包`f`和一个均值序列`sq`两个参数。我们可以看到在`update-seq`函数中定义的行为是假如均值序列中的某一个元素没有重复出现那么就照常更新这个均值，而如果某一个元素重复出现了，那么仅仅更新遍历到的第一个值，余下的相同的值则不做操作。现在可以来观察用重新定义的`new-means`函数来更新均值序列，假如均值序列中存在重复出现的值那么每次只会更新其中一个，我们依然传入`'(0 0)`作为初始均值序列，可以在REPL中观察结果，如下所示：

{% codeblock lang:clojure %}
user> (new-means average
                 (point-groups '(0 0) '(0 1 2 3 4) distance)
                 '(0 0))
(2 0)
{% endcodeblock %}

现在我们可以观察利用重新定义后的`new-means`函数进行聚类的结果。现在不管传入的初始均值序列中是否有重复出现的元素，我们最终都可以得到同样的聚类结果，以`'(0 0)`，`'(0 1)`以及`'(0 2)`为例，我们先来看修改之前的结果：

{% codeblock lang:clojure %}
user> ((k-groups '(0 1 2 3 4) distance average)
       '(0 1))
(([0] [1 2 3 4]) ([0 1] [2 3 4])
user> ((k-groups '(0 1 2 3 4) distance average)
       '(0 0))
(([0 1 2 3 4])
user> ((k-groups '(0 1 2 3 4) distance average)
       '(0 2))
(([0 1] [2 3 4])
{% endcodeblock %}

而使用修改之后的`new-means`函数，利用三种初始均值序列来进行聚类后最终都会收敛到同一个聚类结果：

{% codeblock lang:clojure %}
user> ((k-groups '(0 1 2 3 4) distance average)
       '(0 1))
(([0] [1 2 3 4]) ([0 1] [2 3 4]))
user> ((k-groups '(0 1 2 3 4) distance average)
       '(0 0))
(([0 1 2 3 4]) ([0] [1 2 3 4]) ([0 1] [2 3 4]))
user> ((k-groups '(0 1 2 3 4) distance average)
       '(0 2))
(([0 1] [2 3 4]))
{% endcodeblock %}

重新定义后的`new-means`函数对于含有重复出现元素的初始均值序列表现出来的行为同样可以扩展到初始均值序列中的元素是向量时的情况，如下所示：

{% codeblock lang:clojure %}
user> ((k-groups vector-data vec-distance vec-average)
       '([1 1 1] [1 1 1] [1 1 1]))
(([[1 2 3] [3 2 1] [100 200 300] [300 200 100] [50 50 50]])
 ([[1 2 3] [3 2 1]] [[100 200 300] [300 200 100] [50 50 50]])
 ([[1 2 3] [3 2 1] [50 50 50]] [[100 200 300] [300 200 100]])
 ([[1 2 3] [3 2 1]] [[100 200 300] [300 200 100]] [[50 50 50]]))
{% endcodeblock %}

总的来说，上面例子中实现的`k-cluster`函数和`k-groups`函数展示了如何用纯Clojure代码来实现一个K-means聚类算法。

## 使用clj-ml库来聚类数据

`clj-ml`库是Java机器学习库`weka`的封装，这个库提供了几种聚类算法的实现。现在来演示如何利用`clj-ml`库来建立一个K-means聚类器。

>要在Leiningen项目中使用clj-ml库和Incanter库，我们需要把这两个库的依赖加入到project.clj文件中：<br/>
[cc.artifice/clj-ml "0.4.0"]<br/>
[incanter "1.5.4"]<br/>
对于下面我们将会实现的例子中，我们需要修改文件中名字空间的定义从而引入两个库中提供的函数：<br/>
(ns my-namespace<br/>
&nbsp;&nbsp;(:use [incanter core datasets]<br/>
&nbsp;&nbsp;&nbsp;&nbsp;[clj-ml data clusterers]))

在这一章中使用`clj-ml`库的例子中，我们将会使用`Incanter`库中提供的**Iris**数据集来作为我们的训练数据集。这个数据集本质上是150朵花的数据集，每一朵花的样本数据中，都有四个维度的特征来描述这个样本。其中在数据集中每一个样本的四个特征分别是花瓣的宽度和长度以及花萼的宽度和长度。这一个数据集中花的样本来自三个类别，分别是Virginica，Setosa和Versicolor。这个数据使用$$5 \times 150$$大小的矩阵来描述，其中每一朵花属于的种类放在这个矩阵中的最后一列。

我们可以使用`Incanter`库中的`get-dataset`函数，`sel`函数以及`to-vector`函数从而以向量形式从Iris数据集中提取出所有样本的特征值，如下面代码所示。然后我们可以使用`clj-ml`库中的`make-dataset`函数来将之前得到的特征值向量转化为可供`clj-ml`中聚类函数使用的数据格式。我们还需要向`make-dataset`函数传入一个表示特征向量解构的模板从而可以让处理函数来解析传入的数据集，此外还要传入一个字符串表示这个生成的数据集的名字，需要注意的是一开始得到的特征值向量其实一个向量的向量，其中每一个元素也是一个向量类型，整个数据生成过程如下面代码所示：

{% codeblock lang:clojure %}
(def features [:Sepal.Length
               :Sepal.Width
               :Petal.Length
               :Petal.Width])

(def iris-data (to-vect (sel (get-dataset :iris)
                             :cols features)))

(def iris-dataset
  (make-dataset "iris" features iris-data))
{% endcodeblock %}

我们可以在REPL中打印出上面代码定义的`iris-dataset`变量，从而可以观察到最终得到的训练数据集中的内容，如下所示：

{% codeblock lang:clojure %}
user> iris-dataset
#<ClojureInstances @relation iris
@attribute Sepal.Length numeric
@attribute Sepal.Width numeric
@attribute Petal.Length numeric
@attribute Petal.Width numeric
@data
5.1,3.5,1.4,0.2
4.9,3,1.4,0.2
4.7,3.2,1.3,0.2
...
4.7,3.2,1.3,0.2
6.2,3.4,5.4,2.3
5.9,3,5.1,1.8>
{% endcodeblock %}

我们可以利用`clj-ml.clusterers`名字空间中的`make-clusterer`函数来创建一个聚类器。我们可以将要生成聚类器的类型作为第一个参数传给`make-clusterer`函数，而将要生成聚类器时的一些配置参数以map对象形式作为第二个参数传给`make-clusterer`函数。我们可以使用`clj-ml`库中的`cluster-build`函数来训练给定的聚类器。如下面代码所示，我们使用`make-cluster`函数创建了一个新的聚类器，并且用`:k-means`关键字作为参数来指定了生成的聚类器的类型，然后定义了`train-clusterer`函数来帮助我们利用给定的数据集来训练这个新生成的聚类器：

{% codeblock lang:clojure %}
(def k-means-clusterer
  (make-clusterer :k-means
                  {:number-clusters 3}))

(defn train-clusterer [clusterer dataset]
  (clusterer-build clusterer dataset)
  clusterer)
{% endcodeblock %}

我们可以将用`k-means-clusterer`变量表示的聚类器实例以及用`iris-dataset`变量表示的训练数据集传入`train-clusterer`函数从而观察聚类器训练后的结果，如下代码所示：

{% codeblock lang:clojure %}
user> (train-clusterer k-means-clusterer iris-dataset)
#<SimpleKMeans
kMeans
======

Number of iterations: 6
Within cluster sum of squared errors: 6.982216473785234
Missing values globally replaced with mean/mode

Cluster centroids:
                            Cluster#
Attribute       Full Data          0          1          2
                    (150)       (61)       (50)       (39)
==========================================================
Sepal.Length       5.8433     5.8885      5.006     6.8462
Sepal.Width        3.0573     2.7377      3.428     3.0821
Petal.Length        3.758     4.3967      1.462     5.7026
Petal.Width        1.1993      1.418      0.246     2.079
{% endcodeblock %}

如上面代码的输出结果所示，训练后聚类器的聚类结果中在第一个簇(cluster0)中有61个样本，在第二个簇(cluster1)中有50个样本，而在第三个簇(cluster2)中有39个值。上面的输出结果同样告诉了我们聚类后的三个簇中的每一个簇最后的均值序列的值，也就是最终对给定数据进行聚类之后，每一类中四个特征的平均值。现在我们可以将训练好的聚类器实例传入`clj-ml`库提供的`clusterer-cluster`函数中，从而可以对给定的输入样本数据进行类别预测，如下所示：

{% codeblock lang:clojure %}
user> (clusterer-cluster k-means-clusterer iris-dataset)
#<ClojureInstances @relation 'clustered iris'
@attribute Sepal.Length numeric
@attribute Sepal.Width numeric
@attribute Petal.Length numeric
@attribute Petal.Width numeric
@attribute class {0,1,2}
@data
5.1,3.5,1.4,0.2,1
4.9,3,1.4,0.2,1
4.7,3.2,1.3,0.2,1
...
6.5,3,5.2,2,2
6.2,3.4,5.4,2.3,2
5.9,3,5.1,1.8,0>
{% endcodeblock %}

`clusterer-cluster`函数使用训练好的聚类器实例返回一个新的数据集，在这个新返回的数据集中将会增加一列，表示每一个样本数据在经过聚类之后所分到的类别编号。如上面输出结果所示，这一个新的列中的值为0，1或者2，分别代表了对应这一行的样本属于cluster0，cluster1后者是cluster2。总的来说，`clj-ml`为我们使用聚类算法提供了一套非常易用的框架，并且在上面的例子中我们使用`clj-ml`库创建并训练了一个K-means聚类器。

## 使用层次聚类


