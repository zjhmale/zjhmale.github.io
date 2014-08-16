---
layout: post
title: "clojure macros and metaprogramming"
date: 2013-08-21 09:44:20 +0800
comments: true
categories: clojure fp
---

Clojure是一种lisp的方言，拥有纯正的lisp血统，所以传统lisp的特性，Clojure同样具备。其中一个强大的特性就是**宏(macro)**，一种可以利用元编程之力的特性。相对于其他具有元编程能力的语言(比如ruby)来说，Clojure利用宏来进行meta programming会更加的优雅简洁。在Clojure中，元编程并不只是意味着产生字符串，相反的，这意味着构建一颗新的语法树(S-expressions, 或者lists)，因为lisp中抽象语法树就是用s表达式来描述的。使用宏也意味着Clojure有强大的DSL(domain-specific-language)构建能力。

## 编译器与运行期

Clojure是一个编译型语言，编译器将源程序读入，得到抽象语法树(AST)，然后执行宏扩展。所以宏是在编译器被执行求值的，并且最终生成由宏规定的AST，与普通程序一样，最终被编译成JVM字节码，由JVM在运行期统一执行这些生成的字节码。

Clojure代码可以使用`clojure.core/load`或者`clojure.core/require`进行编译，或者使用构建工具，比如[Leiningen](http://leiningen.org/)或者[Clojure maven plugin](https://github.com/talios/clojure-maven-plugin)，我自己更喜欢用Leiningen，maven从写Java那伙就没怎么用过(可能是一直没做过特别大的项目吧= =)。

## clojure解析器(clojure reader)

Reader只是Parser的一个别称。与其他语言不同，Clojure中的Reader可以扩展语言本身。可以使用`clojure.core/read`或者`clojure.core/read-string`函数将普通的字符串源代码以数据结构的形式暴露给语言本身。

{% codeblock lang:clojure %}
user> (read-string "(if true :truth :false)")
;; here we get back a list that is not evaluated
;= (if true :truth :false)

user> (eval (read-string "(if true :truth :false)"))
;; here we get back a result that is evaluated
;= :true
{% endcodeblock %}

使用Reader得到数据结构之后，就可以对这个结果进行求值啦(这也是为什么在lisp中代码和数据是没有本质区别的)，求值的基本规则如下

* 基础的数据类型(strings integers vectors)会直接对齐本身求值，得到相应的值
* 列表会将第一个元素当做函数，后续的元素作为参数，求值得到结果
* Symbols(使用def定义的)会求值得到其定义时绑定的值

表达式可以以*forms*的形式来进行求值，Forms由如下元素组成

* 函数(Functions)
* 宏(Macros)
* 特殊形式(Special forms)

## 特殊形式(Special Forms)

Reader会使用一些特殊的方式解析一些特殊的forms，这些forms叫做**special forms**，由以下一些元素组成

* .(一个dot，也是一种特殊的form哦)
* new
* set!
* def
* var
* fn*(不使用解构的`fn`)
* if
* case*(`case`的内部实现)
* do
* let*(不使用解构的`let`)
* letfn*(不使用解构的`letfn`)
* clojure.core/import*(`import`)
* quote
* loop*(不使用解构的`loop`)
* recur
* throw, try, catch, finally
* deftype*(deftype的内部实现)
* reify*(`reify`的内部实现)
* monitor-enter, monitor-exit

有一些特殊形式可以直接在代码中使用(比如`do`和`if`)，而另一些特殊形式则是用于构建更易用的接口(比如使用`deftype*`特殊形式来构建`deftype`)

## 牛刀小试

一些语言中有与`if`逻辑原语相反的`unless`表达式。Clojure中并不原生支持，但是可以使用宏轻松实现一个

{% codeblock lang:clojure %}
(defmacro unless
  "Similar to if but negates the condition"
  [condition & forms]
  `(if (not ~condition)
     ~@forms))
{% endcodeblock %}

可以使用`clojure.core/defmacro`函数定义一个宏，上面代码中定义的`unless`宏可以像`if`表达式一样直接使用

{% codeblock lang:clojure %}
(unless (= 1 2)
  "one does not equal two"
  "one equals two. How come?")
{% endcodeblock %}

我们可以使用`clojure.core/macroexpand-1`来看看一个宏是如何展开的

{% codeblock lang:clojure %}
(macroexpand-1 '(unless (= 1 2) true false))
;= (if (clojure.core/not (= 1 2)) true false)
{% endcodeblock %}

下面再看一个利用宏将代码当数据操作的例子

{% codeblock lang:clojure %}
(defmacro postfix-notation
  "I'm too indie for prefix notation"
  [expression]
  (conj (butlast expression) (last expression)))

(postfix-notation (1 1 +))
;= 2

(macroexpand-1 '(postfix-notation (1 1 +)))
;= (+ 1 1)
{% endcodeblock %}

而如果这个使用`defn`来实现的话就会很蛋疼

{% codeblock lang:clojure %}
(defn postfix-notation-fn
  [expression]
  (conj (butlast expression) (last expression)))

(postfix-notation-fn (1 1 +))
;= java.lang.ClassCastException: java.lang.Long cannot be cast to clojure.lang.IFn

(postfix-notation-fn '(1 1 +))
;= (+ 1 1)

(eval (postfix-notation-fn '(1 1 +)))
;= 2
{% endcodeblock %}

可以看到如果定义成函数的话，要将表达式作为参数传入只能先quote住，不然在作为参数传入的过程中，就被求值了，而返回的结果也需要eval之后才能得到最终的求值结果，而宏的话不仅可以传入表达式参数，而且最后返回的只要是一个list，那最终都会自动展开求值并返回结果

当使用宏编程时，有几样工具是必须要掌握的

* Quote(')
* Syntax quote(`)
* Unquote(~)
* Unquote splicing(~@)

## Quote

Quote抑制了表达式的求值作用，只是将代码单纯地当做数据贮存在内存中。可以看一个例子

{% codeblock lang:clojure %}
;; this form is evaluated by calling the clojure.core/+ function
(+ 1 2 3)
;= 6
;; quote supresses evaluation so the + is treated as a regular
;; list element
'(+ 1 2 3)
;= (+ 1 2 3)
`(+ 1 2 3)
;= (clojure.core/+ 1 2 3)
{% endcodeblock %}

quote和syntax quote的主要区别在于syntax quote可以加上名字空间(namespace)，这可以有效地避免名字冲突，另一个使用syntax quote的好处是，在syntax quote中我们还可以使用unquote操作来对某一些特定的部分求值，而不是对印制整一个表达式的求值，在需要的部分我们可以局部唤醒，使用syntax quote可以说是做了一个模板，决定了哪一部分是固定的，而哪一部分是变化的需要求值的。

## Unquote

Unquote可以在syntax quote中强制求值，我们可以稍微改造一下上面的例子看一下如果没有unquote会怎么样

{% codeblock lang:clojure %}
;; incorrect, missing unquote!
(defmacro unless
  [condition & forms]
  `(if (not condition)
     ~@forms))
;= clojure.lang.Compiler$CompilerException: java.lang.RuntimeException: No such var: lazyseq.core/condition, compiling:(/private/var/folders/00/0rkbg5r53cz7zs4gzbp6hc7m0000gn/T/form-init583513466025470119.clj:1:1)

(macroexpand-1 '(unless (= 1 2) true false))
;= (if (clojure.core/not user/condition) true false)
{% endcodeblock %}

可以看到如果没有unqute的话，就会将condition作为外部代码中的一个symbol了，就会从当前名字空间中搜寻这个symbol，所以我们需要用unquote来强制唤醒其对应的值，也就是我们传入的表达式，unquote的表示方法其实只是`clojure.core/unquote`的语法糖。

## Unquote-splicing

有一些需要用到多个forms，正在构建DSLs的时候非常常见，这样的话就需要把每一个form都先quote住，然后再将它们拼接起来，非常的麻烦，但是使用unquote-splicing(`~@`)可以让这一切都异常优雅

依然可以改造之前的例子，看一下没有unquote-splicing会怎么样

{% codeblock lang:clojure %}
(defmacro unless-withoutsplicing
  [condition & forms]
  `(if (not ~condition)
     ~forms))

(macroexpand '(unless-withoutsplicing true
                                     :true
                                     :false))
;= (if (clojure.core/not true) (:true :false))
{% endcodeblock %}

可以看到如果没有unquote-splicing的话，那么多个form并不会组成一个预期希望的形式，而是按照它们在被传值是就被确定的那个样子

{% codeblock lang:clojure %}
user> (defmacro unsplice
        [& coll]
        `(do ~@coll))
;= #'user/unsplice

(macroexpand-1 '(unsplice (def a 1) (def b 2)))
;= (do (def a 1) (def b 2))

(unsplice (def a 1) (def b 2))
;= #'user/b

a
;= 1
b
;= 2

(defmacro unsplice-withoutsplice
  [& coll]
  `(do ~coll))

(macroexpand-1 '(unsplice-withoutsplice (def a 1) (def b 2)))
;= (do ((def a 1) (def b 2)))
{% endcodeblock %}

再通过一个例子可以更加深入的了解到有了unquote-splicing的好处，这样我们就不需要额外的手动去讲多个form从参数中抽离出来，然后再拼接成一个可以被丢出来执行的表达式，我们仅仅只需要使用`~@`这一个宏就行了

和unquote类似unquote-splicing是`clojure.core/unquote-splicing`的语法糖

## Macro Hygiene and gensym

有时候在编写宏的时候可能会需要与外部定义的局部变量交互。但是大部分语言都会将变量的包裹在一层一层的作用域中，而这样的宏也被叫做*unhygienic macros*

先看两篇wiki

* [Variable shadowing](http://en.wikipedia.org/wiki/Variable_shadowing)
* [Hygienic macro](http://en.wikipedia.org/wiki/Hygienic_macro)

首先来说下啥是Variable shadowing，其实这就是正常程序员理解的作用域，也就是说当两个变量名字相同的时候，内层变量拥有执行权从而屏蔽了外层变量。用wiki上的那个例子就可以很好的理解怎么一回事了

{% codeblock lang:lua %}
v = 1 -- a global variable
do
  local v = v+1 -- creates new local that shadows global v
  print(v) --> 2
  do
    local v = v*2 -- another local that shadows
    print(v) --> 4
  end
  print(v) --> 2
end
print(v) --> 1
{% endcodeblock %}

再来看看啥是Hygienic macro，其实就是这个宏可以直接访问外部的symbol，而不会为宏里面的任何symbol建立一块单独的作用域从而屏蔽了外部的symbol

仍然是wiki上的例子，假如是一个unhygienic macro，比如是c中的宏，那么一种常见的情况就是

{% codeblock lang:c %}
#define INCI(i) {int a=0; ++i;}

int main(void)
{
    int a = 0, b = 0;
    INCI(a);
    INCI(b);
    printf("a is now %d, b is now %d\n", a, b);
    return 0;
}
{% endcodeblock %}

上面这段c代码在预编译期会被扩展为如下形式

{% codeblock lang:c %}
int main(void)
{
    int a = 0, b = 0;
    {int a=0; ++a;};
    {int a=0; ++b;};
    printf("a is now %d, b is now %d\n", a, b);
    return 0;
}
{% endcodeblock %}

可以看到预编译展开之后其实最外层作用域的a变量其实是被内部scope的a给屏蔽了，自加操作最终并没有作用到这个变量上，最终执行的结果也不是我们希望看到的

{% codeblock lang:c %}
a is now 1, b is now 1
{% endcodeblock %}

在c中为了解决这个问题最好的方式就是在宏内部不要有和外部作用域重名的变量存在，这在自己玩玩的逗逼小程序里的确make sense，但是你如果是接手其他人留下来的庞大工程，或者是自己几个月前写的东西，你还敢写宏么

{% codeblock lang:c %}
#define INCI(i) {int INCIa=0; ++i;}
int main(void)
{
    int a = 0, b = 0;
    INCI(a);
    INCI(b);
    printf("a is now %d, b is now %d\n", a, b);
    return 0;
}
{% endcodeblock %}

结果就是正常的了

{% codeblock lang:c %}
a is now 1, b is now 1
{% endcodeblock %}

明白了上面两个概念之后，我们再回到Clojure，Clojure中为实现hygienic macros而制定两种约束

* 在syntax quote中的symbol是名字空间限定的，也就是说其中的symbol都是在当前外部名字空间中的
* 如果在宏中想使用和外部名字空间相同的名字而又不想屏蔽外部的symbol，可以使用gensyms机制

## Namespace Qualification Within Syntax Quote

{% codeblock lang:clojure %}
(defmacro yes-no->boolean
  [val]
  `(let [b (= ~val "yes")]
    b))
;= #'user/yes-no->boolean

(macroexpand-1 '(yes-no->boolean "yes"))
;= (clojure.core/let [user/b (clojure.core/= "yes" "yes")] user/b)

(yes-no->boolean "yes")
;= clojure.lang.Compiler$CompilerException: java.lang.RuntimeException: Can't let qualified name: lazyseq.core/b, compiling:(/private/var/folders/00/0rkbg5r53cz7zs4gzbp6hc7m0000gn/T/form-init583513466025470119.clj:1:1)
{% endcodeblock %}

可以看到因为Clojure中的宏可以直接capture到外部名字空间的symbol，所以运行期就出错啦，因为我们在外部根本没有定义这么一个symbol，所以如果我们想使用这个宏就要用到gensyms机制了

## Generated Symbols(gensyms)

自动名字空间生成(Automatic namespace generation)机制在一些情况下是起作用的，但不是所有情况。有时候我们希望在宏的作用域内symbol的名字可以是独一无二的。

独一无二的symbol名字(Unique symbols names)可以使用`clojure.core/gensym`函数生成

{% codeblock lang:clojure %}
(gensym)
;= G__54
(gensym "base")
;= base57
{% endcodeblock %}

`gensym`当然是有语法糖的，在syntax quote中使用`#`就可以自动调用`gensym`

修改上面那个出错的例子

{% codeblock lang:clojure %}
(defmacro yes-no->boolean
  [val]
  `(let [b# (= ~val "yes")]
     b#))
;= #'user/yes-no->boolean

(macroexpand-1 '(yes-no->boolean "yes"))
;= (clojure.core/let [b__148__auto__ (clojure.core/= "yes" "yes")] b__148__auto__)

(yes-no->boolean "yes")
;= true
{% endcodeblock %}

`b__148__auto__`这个名字是由编译器产生的，用于避免屏蔽外部的名字，是一个unique symbol name。

## 宏扩展(Macroexpansions)

在写宏的时候一个很重要的调试技巧就是看看这个宏在编译器展开成什么东西了，以防其在运行期执行时崩溃，能进行宏展开的操作工具基本有下面三种

* `clojure.core/macroexpand-1`
* `clojure.core/macroexpand`
* `clojure.walk/macroexpand-all`

`macroexpand-1`和`macroexpand`之间的区别就是，`macroexpand-1`只会展开一阶宏，假如宏的返回时返回的那个list中还调用了其他的宏，那么是没法用`macroexpand-1`展开的。但是`macroexpand`就可以持续展开宏，直到看不到任何新的宏位置。

当然要展开一个宏，首先要quote住，不然它可以自己先求值了

{% codeblock lang:clojure %}
(macroexpand '(and true false true))
{% endcodeblock %}

使用宏展开我们可以发现`out`其实是一个宏，是利用`if`以及`let*`等special form写成的

{% codeblock lang:clojure %}
user> (macroexpand '(and true false true))
;; formatted for readability
(let* [and__3822__auto__ true]
  (if and__3822__auto__
      (clojure.core/and false true)
      and__3822__auto__))

user> (macroexpand '(and true))
true
{% endcodeblock %}

下面举一个小例子来看`macroexpand`比`macroexpand-1`牛逼在哪里

{% codeblock lang:clojure %}
(defmacro remote-macro
  []
  `(str "cleantha"))

(defmacro current-macro
  []
  `(remote-macro))

(current-macro)
;= "cleantha"

(macroexpand-1 '(current-macro))
;= (lazyseq.core/remote-macro)

(macroexpand '(current-macro)
;= (clojure.core/str cleantha)
{% endcodeblock %}

从上面应该可以知道区别了

## Full Macroexpansion

`macroexpansion-1`和`macroexpand`都没办法展开一个嵌套的form，也就是一个嵌套的宏，所谓的嵌套也就是在一个宏内部有调用了宏，已递归形式嵌套了，那么就只能是用`clojure.walk/macroexpand-all`来展开了，但是这个函数不是在Clojure的核心库中，并且貌似和编译器行为也不太一致。

下面依然使用几个例子说明

{% codeblock lang:clojure %}
user=> (macroexpand-1 '(-> c (+ 3) (* 2)))
(clojure.core/-> (clojure.core/-> c (+ 3)) (* 2))

user=> (macroexpand '(-> c (+ 3) (* 2)))
(* (clojure.core/-> c (+ 3)) 2)

user=> (use 'clojure.walk)
user=> (macroexpand-all '(-> c (+ 3) (* 2)))
(* (+ c 3) 2)

(macroexpand '(.. arm getHand getFinger))
;= (. (. arm getHand) getFinger)

(macroexpand '(-> arm getHand getFinger))
;= (getFinger (clojure.core/-> arm getHand))
{% endcodeblock %}

但是其实`->`这个宏在clojure1.6.0版本中已经可以用`macroexpand`和`macroexpand-1`完全展开了，上面的问题只出现在1.6之前的版本中，所以实际用的时候几个都试试，不用死扣这些细节

下面国外一个老哥回答的挺好的

>macroexpand does not expand macros in subforms, so (-> arm getHand getFinger) expands to (clojure.core/-> (clojure.core/-> arm getHand) getFinger) which expands (because -> is a macro) to (getFinger (clojure.core/-> arm getHand)). The expansion stops here because getFinger is not a macro.

* [出处](http://stackoverflow.com/questions/4304424/clojure-macroexpand)

## Quote 与 Syntax Quote 的不同之处

上面已经提过了两种抑制求值方法的不同之处，这里再总结一下子

* syntax quote会带上名字空间避免名字冲突
* syntax quote中可以使用unquote操作来唤醒特定的求值操作

{% codeblock lang:clojure %}
(defmacro yes-no->boolean-quote
  [val]
  '(let [b# (= ~val "yes")]
     b#))

(macroexpand '(yes-no->boolean-quote "yes"))
;= (let* [b# (= (clojure.core/unquote val) yes)] b#)

(defmacro yes-no->boolean
  [val]
  `(let [b# (= ~val "yes")]
     b#))

(macroexpand '(yes-no->boolean "yes"))
;= (let* [b__1390__auto__ (clojure.core/= yes yes)] b__1390__auto__)
{% endcodeblock %}

此外还有`~'`和`'~`这两个操作，其中`~'`和`gensym`的操作非常相似，就是可以在宏中定义一个局部作用域屏蔽外部名字空间中的symbol，`'~`操作则是可以获取传入参数的形式名字，而不是一个带有名字空间限定的名字，看下面两个例子就能明白了

{% codeblock lang:clojure %}
(defmacro yes-no->boolean-another
  [val]
  `(let [~'b (= ~val "yes")]
     ~'b))

(yes-no->boolean-another "yes")
;= true

(macroexpand-1 '(yes-no->boolean-another "yes"))
;= (clojure.core/let [b (clojure.core/= yes yes)] b)

(defmacro debug [x] `(println "---" '~x ":" ~x))
(defmacro debug2 [x] `(println "---" 'x ":" ~x))

(let [a 10] (debug a))
;= --- a : 10
(let [a 10] (debug2 a))
;= --- lazyseq.core/x : 10
{% endcodeblock %}

## Security Considerations

`clojure.core/read-string`可以执行任意的代码，所以千万不要用这个函数去load任何非信任源处得到的代码。当然可以使用`clojure.core/*read-eval*`变量来控制这种危险的行文。从Clojure1.5开始，`*read-eval*`默认值是`false`。

`*read-eval*`可以在JVM启动时开启或者关闭，可以通过调整JVM参数来实现

{% codeblock %}
-Dclojure.read.eval=false
{% endcodeblock %}

当要从非信任的源中读取源代码时，可以使用`clojure.edn/read-string`，可以来限制执行任意代码的行为，可以做到一定的安全性。`clojure.edn/read-string`实现了[EDN format](https://github.com/edn-format/edn)，是一个用于表示数据结构的Clojure语法子集，在1.5之后引入，与`Datomic`等其他Clojure应用交换数据的一种特定格式。

## Special Forms in Detail

Special forms在Clojure中的使用需要有严格的限定。

* Special form必须是一个列表，并且第一个元素是一个Special name

一个在高阶上下文中的Special name不是一个Special form

{% codeblock lang:clojure %}
user=> do
CompilerException java.lang.RuntimeException: Unable to resolve symbol: do in this context, compiling:(NO_SOURCE_PATH:0:0)
{% endcodeblock %}

Macros也有类似的限定，但是要注意的是，macro在上下文中是有意义的，但是Special name完全不知道这个东西，从上面两个出错信息的内容中也能看到

* Special form 名字不是被名字空间限定的

大多数的special form(除了`clojure.core/import*`)不是用名字空间限定的，读者必须规避掉之前所有symbol都是有名字空间限定的这个概念。

{% codeblock lang:clojure %}
user=> `a
user/a
user=> `do
do
user=> `if
if
user=> `import*
user/import*
{% endcodeblock %}

* Special form会和局部作用域中的名字冲突

永远都不要使用Special name作为局部或者全局变量的名字。

{% codeblock lang:clojure %}
(let [do 1]
  (println do))
;= 1
{% endcodeblock %}

这也包括解构操作

{% codeblock lang:clojure %}
(let [{:keys [do]} {:do 1}]
  (println do))
;= 1
{% endcodeblock %}

关于上面这个解构方法，可以参考下面这个例子

{% codeblock lang:clojure %}
(let [{:keys [cleantha clea]} {:cleantha 3 :clea 1}]
  (println cleantha)
  (println clea))
;= 3
;= 1
{% endcodeblock %}
