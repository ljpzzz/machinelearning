# 刘建平Pinard的博客配套代码

http://www.cnblogs.com/pinard 刘建平Pinard

之前不少朋友反应我博客中的代码都是连续的片段，不好学习，因此这里把文章和代码做一个整理。
代码有部分来源于网络，已加上相关方版权信息。部分为自己原创，已加上我的版权信息。

## 目录

* [机器学习基础与回归算法部分](#2)

* [机器学习分类算法部分](#3)

* [机器学习聚类算法部分](#4)

* [机器学习降维算法部分](#5)

* [机器学习集成学习算法部分](#6)

* [数学统计学部分](#7)

* [机器学习关联算法部分](#8)

* [机器学习推荐算法部分](#9)

* [深度学习算法部分](#10)

* [强化学习算法部分](#1)

## 注意

2016-2017年写的博客使用的python版本是2.7， 2018年因为TensorFlow对Python3的一些要求，所以写博客使用的Python版本是3.6。少部分2016，2017年的博客代码无法找到，重新用Python3.6跑过上传，因此可能会出现和博客中代码稍有不一致的地方，主要涉及到print的语法和range的用法，若遇到问题，稍微修改即可跑通。

<h3 id="1">强化学习文章与代码：:</h3>

|文章 | 代码|
---|---
[强化学习（一）模型基础](https://www.cnblogs.com/pinard/p/9385570.html)| [代码](https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/introduction.py)
[强化学习（二）马尔科夫决策过程(MDP)](https://www.cnblogs.com/pinard/p/9426283.html) | 无
[强化学习（三）用动态规划（DP）求解](https://www.cnblogs.com/pinard/p/9463815.html) | 无
[强化学习（四）用蒙特卡罗法（MC）求解](https://www.cnblogs.com/pinard/p/9492980.html) | 无
[强化学习（五）用时序差分法（TD）求解](https://www.cnblogs.com/pinard/p/9529828.html) | 无
[强化学习（六）时序差分在线控制算法SARSA](https://www.cnblogs.com/pinard/p/9614290.html)  | [代码](https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/sarsa_windy_world.py)

<h3 id="2">机器学习基础与回归算法文章与代码：</h3>

|文章 | 代码|
---|---
[梯度下降（Gradient Descent）小结](https://www.cnblogs.com/pinard/p/5970503.html) | 无
[最小二乘法小结](https://www.cnblogs.com/pinard/p/5976811.html) |无
[交叉验证(Cross Validation)原理小结](https://www.cnblogs.com/pinard/p/5992719.html) | 无
[精确率与召回率，RoC曲线与PR曲线](https://www.cnblogs.com/pinard/p/5993450.html) |无
[线性回归原理小结](https://www.cnblogs.com/pinard/p/6004041.html) |无
[机器学习研究与开发平台的选择](https://www.cnblogs.com/pinard/p/6007200.html) | 无
[scikit-learn 和pandas 基于windows单机机器学习环境的搭建](https://www.cnblogs.com/pinard/p/6013484.html) |无
[用scikit-learn和pandas学习线性回归](https://www.cnblogs.com/pinard/p/6016029.html) |[代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/linear-regression.ipynb)
[Lasso回归算法： 坐标轴下降法与最小角回归法小结](https://www.cnblogs.com/pinard/p/6018889.html) | 无
[用scikit-learn和pandas学习Ridge回归](https://www.cnblogs.com/pinard/p/6023000.html) | [代码1](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/ridge_regression_1.ipynb) [代码2](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/ridge_regression.ipynb)
[scikit-learn 线性回归算法库小结](https://www.cnblogs.com/pinard/p/6026343.html)|无

<h3 id="3">机器学习分类算法文章与代码：</h3>

|文章 | 代码|
---|---
[逻辑回归原理小结](https://www.cnblogs.com/pinard/p/6029432.html) |无
[scikit-learn 逻辑回归类库使用小结](https://www.cnblogs.com/pinard/p/6035872.html) |无
[感知机原理小结](https://www.cnblogs.com/pinard/p/6042320.html) |无
[决策树算法原理(上)](https://www.cnblogs.com/pinard/p/6050306.html) |无
[决策树算法原理(下)](https://www.cnblogs.com/pinard/p/6053344.html)|无
[scikit-learn决策树算法类库使用小结](https://www.cnblogs.com/pinard/p/6056319.html) |[代码1](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/decision_tree_classifier.ipynb) [代码2](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/decision_tree_classifier_1.ipynb)
[K近邻法(KNN)原理小结](https://www.cnblogs.com/pinard/p/6061661.html) |无
[scikit-learn K近邻法类库使用小结](https://www.cnblogs.com/pinard/p/6065607.html) |[代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/knn_classifier.ipynb)
[朴素贝叶斯算法原理小结](https://www.cnblogs.com/pinard/p/6069267.html) |无
[scikit-learn 朴素贝叶斯类库使用小结](https://www.cnblogs.com/pinard/p/6074222.html)| [代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/native_bayes.ipynb)
[最大熵模型原理小结](https://www.cnblogs.com/pinard/p/6093948.html)|无
[支持向量机原理(一) 线性支持向量机](https://www.cnblogs.com/pinard/p/6097604.html)|无
[支持向量机原理(二) 线性支持向量机的软间隔最大化模型](https://www.cnblogs.com/pinard/p/6100722.html)|无
[支持向量机原理(三)线性不可分支持向量机与核函数](https://www.cnblogs.com/pinard/p/6103615.html)|无
[支持向量机原理(四)SMO算法原理](https://www.cnblogs.com/pinard/p/6111471.html)|无
[支持向量机原理(五)线性支持回归](https://www.cnblogs.com/pinard/p/6113120.html)|无
[scikit-learn 支持向量机算法库使用小结](https://www.cnblogs.com/pinard/p/6117515.html)|无
[支持向量机高斯核调参小结](https://www.cnblogs.com/pinard/p/6126077.html) | [代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/svm_classifier.ipynb)

<h3 id="7">数学统计学文章与代码：</h3>

|文章 | 代码|
---|---
[机器学习算法的随机数据生成](https://www.cnblogs.com/pinard/p/6047802.html) | [代码](https://github.com/ljpzzz/machinelearning/blob/master/mathematics/random_data_generation.ipynb)

<h3 id="6">机器学习集成学习文章与代码：</h3>

|文章 | 代码|
---|---
[集成学习原理小结](https://www.cnblogs.com/pinard/p/6131423.html) | 无
[集成学习之Adaboost算法原理小结](https://www.cnblogs.com/pinard/p/6133937.html) | 无
[scikit-learn Adaboost类库使用小结](https://www.cnblogs.com/pinard/p/6136914.html) | [代码](https://github.com/ljpzzz/machinelearning/blob/master/ensemble-learning/adaboost-classifier.ipynb)
[梯度提升树(GBDT)原理小结](https://www.cnblogs.com/pinard/p/6140514.html) | 无
[scikit-learn 梯度提升树(GBDT)调参小结](https://www.cnblogs.com/pinard/p/6143927.html)| [代码](https://github.com/ljpzzz/machinelearning/blob/master/ensemble-learning/gbdt_classifier.ipynb)
[Bagging与随机森林算法原理小结](https://www.cnblogs.com/pinard/p/6156009.html) | 无
[scikit-learn随机森林调参小结](https://www.cnblogs.com/pinard/p/6160412.html) |  [代码](https://github.com/ljpzzz/machinelearning/blob/master/ensemble-learning/random_forest_classifier.ipynb)

<h3 id="4">机器学习聚类算法文章与代码：</h3>

|文章 | 代码|
---|---
[K-Means聚类算法原理](https://www.cnblogs.com/pinard/p/6164214.html)|无
[用scikit-learn学习K-Means聚类](https://www.cnblogs.com/pinard/p/6169370.html) | [代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/kmeans_cluster.ipynb)
[BIRCH聚类算法原理](https://www.cnblogs.com/pinard/p/6179132.html)|无
[用scikit-learn学习BIRCH聚类](https://www.cnblogs.com/pinard/p/6200579.html) | [代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/birch_cluster.ipynb)
[DBSCAN密度聚类算法](https://www.cnblogs.com/pinard/p/6208966.html)|无
[用scikit-learn学习DBSCAN聚类](https://www.cnblogs.com/pinard/p/6217852.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/dbscan_cluster.ipynb)
[谱聚类（spectral clustering）原理总结](https://www.cnblogs.com/pinard/p/6221564.html) |无
[用scikit-learn学习谱聚类](https://www.cnblogs.com/pinard/p/6235920.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/spectral_cluster.ipynb)

<h3 id="5">机器学习降维算法文章与代码：</h3>

|文章 | 代码|
---|---
[主成分分析（PCA）原理总结](https://www.cnblogs.com/pinard/p/6239403.html)|无
[用scikit-learn学习主成分分析(PCA)](https://www.cnblogs.com/pinard/p/6243025.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/pca.ipynb)
[线性判别分析LDA原理总结](https://www.cnblogs.com/pinard/p/6244265.html)|无
[用scikit-learn进行LDA降维](https://www.cnblogs.com/pinard/p/6249328.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/lda.ipynb)
[奇异值分解(SVD)原理与在降维中的应用](https://www.cnblogs.com/pinard/p/6251584.html)|无
[局部线性嵌入(LLE)原理总结](https://www.cnblogs.com/pinard/p/6266408.html)|无
[用scikit-learn研究局部线性嵌入(LLE)](https://www.cnblogs.com/pinard/p/6273377.html) |[代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/lle.ipynb)

<h3 id="8">机器学习关联算法文章与代码：</h3>

|文章 | 代码|
---|---
[典型关联分析(CCA)原理总结](https://www.cnblogs.com/pinard/p/6288716.html)|无
[Apriori算法原理总结](https://www.cnblogs.com/pinard/p/6293298.html)|无
[FP Tree算法原理总结](https://www.cnblogs.com/pinard/p/6307064.html)|无
[PrefixSpan算法原理总结](https://www.cnblogs.com/pinard/p/6323182.html)|无
[用Spark学习FP Tree算法和PrefixSpan算法](https://www.cnblogs.com/pinard/p/6340162.html)| [代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/fp_tree_prefixspan.ipynb)
[日志和告警数据挖掘经验谈](https://www.cnblogs.com/pinard/p/6039099.html) | 无

License MIT.
