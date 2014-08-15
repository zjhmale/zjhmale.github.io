---
layout: post
title: "laziness in clojure"
date: 2013-07-19 09:44:20 +0800
comments: true
categories: clojure fp
---

## lazy evaluation

首先申明，`clojure`并不是一个`惰性语言`，关于惰性可以参考[wiki](http://en.wikipedia.org/wiki/Lazy_evaluation)

惰性计算是`call-by-need`的，也就是表达式的值只有在上下文中用到的时候才会被求值。非惰性的语言叫做`eager language`，一个典型的非惰性语言也可以表现出一些惰性的性质，比如在`if`操作中

{% codeblock lang:ruby %}
if a then b else c
{% endcodeblock %}

首先会对表达式`a`求值，然后加入求值得到的结果是`true`，那么就会对表达式`b`求值，否则就会对`c`求值。也就是说表达式`b`和`c`永远只有一个能被求值。相反的，还有一种在非惰性语言中很常见的情况就是

{% codeblock lang:ruby %}
define f(x, y) = 2 * x
set k = f(d, e)
{% endcodeblock %}

当需要用到k的值的时候，表达式`d`和`e`都会被求值，即使表示`e`的值永远不会被用到

{% codeblock lang:ruby %}
define g(a, b, c) = if a then b else c
l = g(h, i, j)
{% endcodeblock %}

表达式`i`和`j`依然会都被求值

只有在

{% codeblock lang:ruby %}
l' = if h then i else j
{% endcodeblock %}

`i`和`j`才会只有其中一个能被求值

可以看到如果没有惰性求值的话我们将会有多少多余的计算，但是如果把惰性计算发挥到极致，像haskell那样，以至于影响到了语言的执行顺序甚至是执行效率，那有点过了，这里没有抨击haskell的意思，haskell很严谨，这也是这个教派追求的东西，所以世间没有银弹嗯。当然惰性计算除了可以减少很多不必要的计算之外，最牛逼的特性就是你可以存储一个无限大的数据结构，这在`clojure`中也是司空见惯的，下面以人人都会的`python`为例。

* 在python2.x中，`range()`这个函数是非惰性的

{% codeblock lang:python %}
r = range(10)
print r
# => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print r[3]
# => 3
{% endcodeblock %}

* 在python3.x中，`range()`这个函数被修改为惰性的了

{% codeblock lang:python %}
r = range(10)
print(r)
# => range(0, 10)
print(r[3])
# => 3
{% endcodeblock %}

当然在py2中也可通过一些比较`hack`的方式得到一个惰性序列

{% codeblock lang:python %}
list = range(10)
iterator = iter(list)
print list
# => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print iterator
# => <listiterator object at 0xf7e8dd4c>
print iterator.next()
# => 0
{% endcodeblock %}

可以看到这个无限大的数据结构其实就是一个`lazy sequence`，一个惰性序列可以认为是扔出来了一个`object`，从这个惰性序列中我们可以取到我们需要得到值，这些值只有在我们需要，我们去拿的时候才会被计算出来，从上面的例子中也可以看到如果没有惰性序列，那么一个序列中的所有元素都要先被求值一次然后放置到内存中，这对于内存资源是一种浪费

## 惰性序列

虽然`clojure`不是惰性语言，但是`clojure`支持惰性序列。就像上面提到的那样，惰性序列主要有以下两个牛逼的特性

* 它们可以以无穷大的形式存在于内存中
* 一个序列的任何元素直到要被拿出来做其他计算使用时才会被求值，不然就一直存在于内存中

## 构造惰性序列

惰性序列是利用函数构造的。我们既可以用`clojure.core/lazy-seq`这样的宏来产生一个惰性序列，或者直接使用能产生惰性序列的函数。

{% codeblock lang:clojure %}
(defn uuid-seq
  []
  (lazy-seq
   (cons (str (UUID/randomUUID))
         (uuid-seq))))
{% endcodeblock %}

`UUID`即`UniversallyUniqueIdentifier`表示一种全局唯一的标识，`uuid-seq`这个函数利用递归深度嵌套了一个惰性序列。另一个例子

{% codeblock lang:clojure %}
(defn fib-seq
  ([]
     (fib-seq 0 1))
  ([a b]
     (lazy-seq
      (cons b (fib-seq b (+ a b))))))
{% endcodeblock %}

`fib-seq`函数用来产生一个惰性的`fibonacci`数列。上面那两个例子都是用了`clojure.core/cons`函数来将一个元素插入到一个序列的头部。然后这个序列被转换为惰性的。

虽然惰性序列是无限的，但是我们可以从中选取我们需要的元素

{% codeblock lang:clojure %}
(take 3 (uuid-seq))
;= ("8da1b70e-7d4d-4972-b4af-48ed248c5568" "b0bc5c2c-f5ff-4733-b3ce-b0499a1a0ccc" "26d52a65-cde5-4d57-bf7f-97e3440fb3a5")

(take 10 (fib-seq))
l= (1 1 2 3 5 8 13 21 34 55)

(take 20 (fib-seq))
;= (1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181 6765)
{% endcodeblock %}

## 唤醒惰性序列(强制求值)

如果想强制唤醒一个惰性序列，可以使用`clojure.core/dorun`或者`clojure.core/doall`，这两个函数的区别在于`dorun`不会返回计算的结果，一般用来做有副作用的操作，而`doall`会返回求值的结果。

{% codeblock lang:clojure %}
(class (map println [1 2 3]))
;=clojure.lang.LazySeq

(map println [1 2 3])
;=(1
;2
;3
;nil nil nil)
{% endcodeblock %}

可以看到map返回的就是一个惰性序列，如果没有额外的操作，副作用的操作是会被包含在这个惰性序列中，而不是将副作用作用在外部环境，这的确符合函数式的思想，但是这不符合一些从其他语言入门编程的人的即有三观。

{% codeblock lang:clojure %}
(dorun (map inc [1 2 3 4]))
;= nil

(doall (map inc [1 2 3 4]))
;= (2 3 4 5)
{% endcodeblock %}

上面的这个例子可能还不够深刻

{% codeblock lang:clojure %}
(doall (map println [1 2 3]))
;1
;2
;3
;=(nil nil nil)

(dorun (map println [1 2 3]))
;1
;2
;3
;=nil

(doall (map #(println "hi" %) ["mum" "dad" "sister"]))
;1
;2
;3
;=(nil nil nil)

(dorun (map #(println "hi" %) ["mum" "dad" "sister"]))
;hi mum
;hi dad
;hi sister
;=nil
{% endcodeblock %}

上面两组例子应该已经很好的讲清楚了两个函数的异同了，当用于产生惰性序列的函数中存在副作用(side effect)，那么就可以使用这两个函数来符合一般人三观地执行副作用，不同的是`doall`会返回最终的惰性序列，但是`dorun`始终返回`nil`。

## 用来产生惰性序列的函数操作

`clojure.core`中常见的返回惰性序列的函数有

* `map`
* `filter`
* `remove`
* `range`
* `take`
* `take-while`
* `drop`
* `drop-while`

可以看一个取数的简单例子

{% codeblock lang:clojure %}
(take 10 (filter even? (range 0 100)))
;= (0 2 4 6 8 10 12 14 16 18)
{% endcodeblock %}

在`clojure.core`中还有一些专门用来产生惰性序列的函数

* `repeat`
* `iterate`
* `cycle`

例如

{% codeblock lang:clojure %}
(take 3 (repeat "ha"))
;= ("ha" "ha" "ha")

(take 5 (repeat "ha"))
;= ("ha" "ha" "ha" "ha" "ha")

(take 3 (cycle [1 2 3 4 5]))
;= (1 2 3)

(take 10 (cycle [1 2 3 4 5]))
;= (1 2 3 4 5 1 2 3 4 5)

(take 3 (iterate (partial + 1) 1))
;= (1 2 3)

(take 5 (iterate (partial + 1) 1))
;= (1 2 3 4 5)
{% endcodeblock %}

## 惰性序列分块

实现惰性序列有两种基本的策略思想

* 对于惰性序列中的元素一个一个唤醒(one-by-one)
* 对于惰性序列中的元素分组唤醒(chunks, batches)

在`clojure` 1.1+中，惰性序列是分块的，也就是要求值时是批量唤醒。

例如下面这个例子

{% codeblock lang:clojure %}
(take 10 (range 1 1000000000000))
{% endcodeblock %}

如果是一个一个地唤醒元素求值，那么急需要进行10次唤醒元素的操作，而如果是分批唤醒元素的话，那么就可以一次唤醒操作就获得前10个元素的值，因为`clojure`中一次唤醒最多可以唤醒32个元素(32 elements a time)，这种做法减少了唤醒操作的次数，而且对于一般的工作场景，还加快了惰性序列唤醒操作的执行效率。
