---
layout: post
title: "destructure in clojure"
date: 2013-05-13 09:44:20 +0800
comments: true
categories: clojure fp
---

>Clojure是一个漂亮优雅的语言，就像是一个美丽的少女，让我着迷

解构(destructure)就是一个让Clojure变得漂亮的一个特性，使用解构可以写出极为简洁优雅的代码

## 什么是解构

>Clojure 支持抽象数据结构绑定，通常也称作解构，这在let绑定，函数参数绑定，或者能展开为let绑定或者函数绑定的宏中都是非常常见的用法

## Destructure Vector

一个解构最简单的例子就是对一个向量赋值

{% codeblock lang:clojure %}
user=> (def point [5 7])
#'user/point

user=> (let [[x y] point]
         (println "x:" x "y:" y))
x: 5 y: 7
{% endcodeblock %}

解构还可以把某一些现在不关心的值都先放在一起处理，可以只是先把前几个关心的值解构出来

{% codeblock lang:clojure %}
user=> (def indexes [1 2 3])
#'user/indexes

user=> (let [[x & more] indexes]
         (println "x:" x "more:" more))
x: 1 more: (2 3)
{% endcodeblock %}

我们还可以使用`:as`来绑定整一个向量

{% codeblock lang:clojure %}
user=> (def indexes [1 2 3])
#'user/indexes

user=> (let [[x & more :as full-list] indexes]
         (println "x:" x "more:" more "full list:" full-list))
x: 1 more: (2 3) full list: [1 2 3]
{% endcodeblock %}

## Destructure Map

解构向量只是很简单的一部分，最常用的还是用来解构映射表(`map`)

{% codeblock lang:clojure %}
user=> (def point {:x 5 :y 7})
#'user/point

user=> (let [{the-x :x the-y :y} point]
         (println "x:" the-x "y:" the-y))
x: 5 y: 7
{% endcodeblock %}

当然我们也可以去掉上面那个例子中`let`内部局部绑定的名字

{% codeblock lang:clojure %}
user=> (def point {:x 5 :y 7})
#'user/point

user=> (let [{x :x y :y} point]
         (println "x:" x "y:" y))
x: 5 y: 7
{% endcodeblock %}

但是假如你需要解构的键超过两个，甚至十多个，而且假如键的长度不止一个字符，那么像上面那样写岂不是很蛋疼，所以Clojure还提供了一种更优雅的解决方案，可以减少一半的工作量

{% codeblock lang:clojure %}
user=> (def point {:x 5 :y 7})
#'user/point

user=> (let [{:keys [x y]} point]
         (println "x:" x "y:" y))
x: 5 y: 7
{% endcodeblock %}

所以可以看到这种初看很怪异的解构写法，和之前的例子是类似的功能，只不过可以让我们不需要把重复的名字敲两遍

同样在解构`map`的时候也可以像解构`vector`一样，通过使用`:as`从而得到整一个要解构的`map`

{% codeblock lang:clojure %}
user=> (def point {:x 5 :y 7})
#'user/point

user=> (let [{:keys [x y] :as the-point} point]
         (println "x:" x "y:" y "point:" the-point))
x: 5 y: 7 point: {:x 5, :y 7}
{% endcodeblock %}

与`:as`对应的是，我们可以使用`:or`来设置解构的默认值，也就是说如果传入的`map`没有对应的解构值，那么我们在上下文中就使用`:or`指定的默认值

{% codeblock lang:clojure %}
user=> (def point {:y 7})
#'user/point
 
user=> (let [{:keys [x y] :or {x 0 y 0}} point]
         (println "x:" x "y:" y))
x: 0 y: 7
{% endcodeblock %}

同样，你也可以使用解构来拆解嵌套的`map`结构

{% codeblock lang:clojure %}
user=> (def book {:name "SICP" :details {:pages 657 :isbn-10 "0262011530"}})
#'user/book

user=> (let [{name :name {:keys [pages isbn-10]} :details} book]
         (println "name:" name "pages:" pages "isbn-10:" isbn-10))
name: SICP pages: 657 isbn-10: 0262011530
{% endcodeblock %}

`map`和`vector`在Clojure内部都是一样的抽象数据结构Sequence，都是序列，所以一般`map`和`vector`的操作都是类似的，所以我们也可以解构一个嵌套的`vector`

{% codeblock lang:clojure %}
user=> (def numbers [[1 2][3 4]])
#'user/numbers

user=> (let [[[a b][c d]] numbers]
         (println "a:" a "b:" b "c:" c "d:" d))
a: 1 b: 2 c: 3 d: 4
{% endcodeblock %}

当然如果是`map`和`vector`嵌套在一起了，也可以轻松解构

{% codeblock lang:clojure %}
user=> (def golfer {:name "Jim" :scores [3 5 4 5]})
#'user/golfer

user=> (let [{name :name [hole1 hole2] :scores} golfer] 
         (println "name:" name "hole1:" hole1 "hole2:" hole2))
name: Jim hole1: 3 hole2: 5
{% endcodeblock %}

## Destructure in Function

Clojure函数中参数传递时其实就是使用了隐式的`let`绑定，所以上面提到的所有解构技巧，都可以使用在Clojure函数的参数传递上

我们可以将上文中最后一个例子应用到函数传参中

{% codeblock lang:clojure %}
user=> (defn print-status [{name :name [hole1 hole2] :scores}] 
         (println "name:" name "hole1:" hole1 "hole2:" hole2))
#'user/print-status

user=> (print-status {:name "Jim" :scores [3 5 4 5]})
name: Jim hole1: 3 hole2: 5
{% endcodeblock %}

再看一些其他的例子

{% codeblock lang:clojure %}
;; Return the first element of a collection
(defn my-first
  [[first-thing]] ; Notice that first-thing is within a vector
  first-thing)

(my-first ["oven" "bike" "waraxe"])
; => "oven"

(defn chooser
  [[first-choice second-choice & unimportant-choices]]
  (println (str "Your first choice is: " first-choice))
  (println (str "Your second choice is: " second-choice))
  (println (str "We're ignoring the rest of your choices. "
                "Here they are in case you need to cry over them: "
                (clojure.string/join ", " unimportant-choices))))
(chooser ["Marmalade", "Handsome Jack", "Pigpen", "Aquaman"])
; => 
; Your first choice is: Marmalade
; Your second choice is: Handsome Jack
; We're ignoring the rest of your choices. Here they are in case \
; you need to cry over them: Pigpen, Aquaman

(defn announce-treasure-location
  [{lat :lat lng :lng}]
  (println (str "Treasure lat: " lat))
  (println (str "Treasure lng: " lng)))
(announce-treasure-location {:lat 28.22 :lng 81.33})
; =>
; Treasure lat: 28.22
; Treasure lng: 81.33

;; Works the same as above.
(defn announce-treasure-location
  [{:keys [lat lng]}]
  (println (str "Treasure lat: " lat))
  (println (str "Treasure lng: " lng)))

;; Works the same as above.
(defn receive-treasure-location
  [{:keys [lat lng] :as treasure-location}]
  (println (str "Treasure lat: " lat))
  (println (str "Treasure lng: " lng))

  ;; One would assume that this would put in new coordinates for your ship
  (println treasure-location))

(receive-treasure-location {:lat 3 :lng 33})
; =>
; Treasure lat: 3
; Treasure lng: 33
; {:lat 3, :lng 33}
{% endcodeblock %}

更多细节可以参考官方文档[special forms](http://clojure.org/special_forms)
