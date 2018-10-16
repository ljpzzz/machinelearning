# 刘建平Pinard的博客配套代码

http://www.cnblogs.com/pinard 刘建平Pinard

之前不少朋友反应我博客中的代码都是连续的片段，不好学习，因此这里把文章和代码做一个整理。
代码有部分来源于网络，已加上相关方版权信息。部分为自己原创，已加上我的版权信息。

## 目录

* [机器学习基础与回归算法](#2)

* [机器学习分类算法](#3)

* [机器学习聚类算法](#4)

* [机器学习降维算法](#5)

* [机器学习集成学习算法](#6)

* [数学统计学](#7)

* [机器学习关联算法](#8)

* [机器学习推荐算法](#9)

* [深度学习算法](#10)

* [自然语言处理算法](#11)

* [强化学习算法](#1)

* [特征工程与算法落地](#12)

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
[强化学习（七）时序差分离线控制算法Q-Learning](https://www.cnblogs.com/pinard/p/9669263.html)  | [代码](https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/q_learning_windy_world.py)
[强化学习（八）价值函数的近似表示与Deep Q-Learning](https://www.cnblogs.com/pinard/p/9714655.html)  | [代码](https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/dqn.py)
[强化学习（九）Deep Q-Learning进阶之Nature DQN](https://www.cnblogs.com/pinard/p/9756075.html)  | [代码](https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/nature_dqn.py)
[强化学习（十）Double DQN (DDQN)](https://www.cnblogs.com/pinard/p/9778063.html)  | [代码](https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/ddqn.py)
[强化学习(十一) Prioritized Replay DQN](https://www.cnblogs.com/pinard/p/9797695.html)  | [代码](https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/ddqn_prioritised_replay.py)

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
[异常点检测算法小结](https://www.cnblogs.com/pinard/p/9314198.html)|无

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
[MCMC(一)蒙特卡罗方法](https://www.cnblogs.com/pinard/p/6625739.html)|无
[MCMC(二)马尔科夫链](https://www.cnblogs.com/pinard/p/6632399.html)| [代码](https://github.com/ljpzzz/machinelearning/blob/master/mathematics/mcmc_2.ipynb)
[MCMC(三)MCMC采样和M-H采样](https://www.cnblogs.com/pinard/p/6638955.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/mathematics/mcmc_3_4.ipynb)
[MCMC(四)Gibbs采样](https://www.cnblogs.com/pinard/p/6645766.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/mathematics/mcmc_3_4.ipynb)

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

<h3 id="9">机器学习推荐算法文章与代码：</h3>

|文章 | 代码|
---|---
[协同过滤推荐算法总结](https://www.cnblogs.com/pinard/p/6349233.html)|无
[矩阵分解在协同过滤推荐算法中的应用](https://www.cnblogs.com/pinard/p/6351319.html)|无
[SimRank协同过滤推荐算法](https://www.cnblogs.com/pinard/p/6362647.html)|无
[用Spark学习矩阵分解推荐算法](https://www.cnblogs.com/pinard/p/6364932.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/matrix_factorization.ipynb)
[分解机(Factorization Machines)推荐算法原理](https://www.cnblogs.com/pinard/p/6370127.html)|无
[贝叶斯个性化排序(BPR)算法小结](https://www.cnblogs.com/pinard/p/9128682.html)|无
[用tensorflow学习贝叶斯个性化排序(BPR)](https://www.cnblogs.com/pinard/p/9163481.html)| [代码](https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/bpr.ipynb)

<h3 id="10">深度学习算法文章与代码：</h3>

|文章 | 代码|
---|---
[深度神经网络（DNN）模型与前向传播算法](https://www.cnblogs.com/pinard/p/6418668.html)|无
[深度神经网络（DNN）反向传播算法(BP)](https://www.cnblogs.com/pinard/p/6422831.html)|无
[深度神经网络（DNN）损失函数和激活函数的选择](https://www.cnblogs.com/pinard/p/6437495.html)|无
[深度神经网络（DNN）的正则化](https://www.cnblogs.com/pinard/p/6472666.html)|无
[卷积神经网络(CNN)模型结构](https://www.cnblogs.com/pinard/p/6483207.html)|无
[卷积神经网络(CNN)前向传播算法](https://www.cnblogs.com/pinard/p/6489633.html)|无
[卷积神经网络(CNN)反向传播算法](https://www.cnblogs.com/pinard/p/6494810.html)|无
[循环神经网络(RNN)模型与前向反向传播算法](https://www.cnblogs.com/pinard/p/6509630.html)|无
[LSTM模型与前向反向传播算法](https://www.cnblogs.com/pinard/p/6519110.html)|无
[受限玻尔兹曼机（RBM）原理总结](https://www.cnblogs.com/pinard/p/6530523.html)|无

<h3 id="11">自然语言处理文章与代码：</h3>

|文章 | 代码|
---|---
[文本挖掘的分词原理](https://www.cnblogs.com/pinard/p/6677078.html)|无
[文本挖掘预处理之向量化与Hash Trick](https://www.cnblogs.com/pinard/p/6688348.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/hash_trick.ipynb)
[文本挖掘预处理之TF-IDF](https://www.cnblogs.com/pinard/p/6693230.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/tf-idf.ipynb)
[中文文本挖掘预处理流程总结](https://www.cnblogs.com/pinard/p/6744056.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/chinese_digging.ipynb)
[英文文本挖掘预处理流程总结](https://www.cnblogs.com/pinard/p/6756534.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/english_digging.ipynb)
[文本主题模型之潜在语义索引(LSI)](https://www.cnblogs.com/pinard/p/6805861.html)|无
[文本主题模型之非负矩阵分解(NMF)](https://www.cnblogs.com/pinard/p/6812011.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/nmf.ipynb)
[文本主题模型之LDA(一) LDA基础](https://www.cnblogs.com/pinard/p/6831308.html)|无
[文本主题模型之LDA(二) LDA求解之Gibbs采样算法](https://www.cnblogs.com/pinard/p/6867828.html)|无
[文本主题模型之LDA(三) LDA求解之变分推断EM算法](https://www.cnblogs.com/pinard/p/6873703.html)|无
[用scikit-learn学习LDA主题模型](https://www.cnblogs.com/pinard/p/6908150.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/lda.ipynb)
[EM算法原理总结](https://www.cnblogs.com/pinard/p/6912636.html)|无
[隐马尔科夫模型HMM（一）HMM模型](https://www.cnblogs.com/pinard/p/6945257.html)|无
[隐马尔科夫模型HMM（二）前向后向算法评估观察序列概率](https://www.cnblogs.com/pinard/p/6955871.html)|无
[隐马尔科夫模型HMM（三）鲍姆-韦尔奇算法求解HMM参数](https://www.cnblogs.com/pinard/p/6972299.html)|无
[隐马尔科夫模型HMM（四）维特比算法解码隐藏状态序列](https://www.cnblogs.com/pinard/p/6991852.html)|无
[用hmmlearn学习隐马尔科夫模型HMM](https://www.cnblogs.com/pinard/p/7001397.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/hmm.ipynb)
[条件随机场CRF(一)从随机场到线性链条件随机场](https://www.cnblogs.com/pinard/p/7048333.html)|无
[条件随机场CRF(二) 前向后向算法评估标记序列概率](https://www.cnblogs.com/pinard/p/7055072.html)|无
[条件随机场CRF(三) 模型学习与维特比算法解码](https://www.cnblogs.com/pinard/p/7068574.html)|无
[word2vec原理(一) CBOW与Skip-Gram模型基础](https://www.cnblogs.com/pinard/p/7160330.html)|无
[word2vec原理(二) 基于Hierarchical Softmax的模型](https://www.cnblogs.com/pinard/p/7243513.html)|无
[word2vec原理(三) 基于Negative Sampling的模型](https://www.cnblogs.com/pinard/p/7249903.html)|无
[用gensim学习word2vec](https://www.cnblogs.com/pinard/p/7278324.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/word2vec.ipynb)

<h3 id="12">特征工程与算法落地文章与代码：</h3>

|文章 | 代码|
---|---
[特征工程之特征选择](https://www.cnblogs.com/pinard/p/9032759.html)|无
[特征工程之特征表达](https://www.cnblogs.com/pinard/p/9061549.html)|无
[特征工程之特征预处理](https://www.cnblogs.com/pinard/p/9093890.html)|无
[用PMML实现机器学习模型的跨平台上线](https://www.cnblogs.com/pinard/p/9220199.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/model-in-product/sklearn-jpmml)
[tensorflow机器学习模型的跨平台上线](https://www.cnblogs.com/pinard/p/9251296.html)|[代码](https://github.com/ljpzzz/machinelearning/blob/master/model-in-product/tensorflow-java)

License MIT.
