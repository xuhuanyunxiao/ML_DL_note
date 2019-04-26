[TOC]

# 第一章 算法对比

##（一）优缺点及改进措施

| 序号 | 算法     | C&P[^1] | 优点                                                         | 缺点                                                         | 改进措施                                                     | 应用领域                                                     |
| ---- | :------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 01   | SVM      | C/P     | 1 解决小样本问题；<br/>2 解决非线性问题；<br/>3 无局部极小值问题；<br/>4 可以很好处理高维数据；<br/>5 泛化能力比较强 | 1 对核函数的高维映射解释能力不强，尤其是径向基函数；<br/>2 对缺失数据敏感 |                                                              | 1 文本分类；<br/>2 图像识别；<br/>3 二分类                   |
| 02   | LR       |         |                                                              |                                                              |                                                              |                                                              |
| 03   | DT       | C/P     | 1 易于理解和解释，可以可视化分析，容易提取规则；<br/>2 可以同时处理标称型和数值型数据； | 1 对缺失数据处理较为困难；<br/>2 容易出现过拟合问题；<br/>3 忽略数据集中属性之间的相互关联；<br/>4 ID3算法计算信息增益时，容易偏向数值较多的特征 | 1 对决策树进行剪枝。交叉验证或加入正则项；<br/>2 使用基于决策树的combination 算法，如RF、Bagging算法 | 1 企业实践管理；<br/>2 企业决策管理；<br/>3 决策过程         |
| 04   | NB       |         |                                                              |                                                              |                                                              |                                                              |
| 05   | KNN      | C       | 1 是一种在线技术，可以直接加入数据，而不需要重新计算模型；<br/>2 理论简单，容易实现 | 1 样本容量越大，则计算量越大；<br/>2 样本不平衡时，预测偏差较大;<br/>3 每次分类，都会重新进行一次全局运算；<br/>4 K值的选择 |                                                              | 1 文本分类；<br/>2 模式识别；<br/>3 聚类分析；<br/>4 多分类领域 |
| 06   | 熵模型   | C       |                                                              |                                                              |                                                              |                                                              |
| 07   | RF       |         |                                                              |                                                              |                                                              |                                                              |
| 08   | GDBT     |         |                                                              |                                                              |                                                              |                                                              |
| 09   | XGBoost  |         |                                                              |                                                              |                                                              |                                                              |
| 10   | Adaboost |         |                                                              |                                                              |                                                              |                                                              |
|      |          |         |                                                              |                                                              |                                                              |                                                              |



## （二）算法应用领域









# 第二章 单一模型

## （一） SVM



## （二） Logistics Regression （LR）



## （三） Decision Tree（DT）



## （四） Naive Bayes （NB）





## （五） KNN

### 1 算法原理

> **知识点：**





### 2 应用操作

#### 1） Python--sklearn

- 语法：
  > KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)
  > - weights：uniform（一样）、distance（以距离判断）
  > - algorithm ：ball_tree（use BallTree）、kd_tree（use KDTree）、brute（use a brute-force search）、auto（自动决定最好的算法）
- 方法：
  - kneighbors([X, n_neighbors, return_distance])	
    
    > Finds the K-neighbors of a point.
  - kneighbors_graph([X, n_neighbors, mode])	
    
    > Computes the (weighted) graph of k-Neighbors for points in X
- 见文件：plot_voting_decision_regions.py
```python
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsClassifier
>>> neigh = KNeighborsClassifier(n_neighbors=3)
>>> neigh.fit(X, y) 
KNeighborsClassifier(...)
>>> print(neigh.predict([[1.1]]))
[0]
>>> print(neigh.predict_proba([[0.9]]))
[[ 0.66666667  0.33333333]]
```
```python
>>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(n_neighbors=1)
>>> neigh.fit(samples) 
NearestNeighbors(algorithm='auto', leaf_size=30, ...)
>>> print(neigh.kneighbors([[1., 1., 1.]])) 
(array([[ 0.5]]), array([[2]]...))

>>> X = [[0., 1., 0.], [1., 0., 1.]]
>>> neigh.kneighbors(X, return_distance=False) 
array([[1],
       [2]]...)
```
```python
>>> X = [[0], [3], [1]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(n_neighbors=2)
>>> neigh.fit(X) 
NearestNeighbors(algorithm='auto', leaf_size=30, ...)
>>> A = neigh.kneighbors_graph(X)
>>> A.toarray()
array([[ 1.,  0.,  1.],
       [ 0.,  1.,  1.],
       [ 1.,  0.,  1.]])
```





## （六） 最大熵模型







# 第三章 集成算法

> **Ensemble Algorithms: **集成算法。多分类器系统。
>
> **combination model: ** 组合模型

集成方法是由多个较弱的模型集成模型组，一般的弱分类器可以是 **DT, SVM, NN, KNN** 等构成。其中的模型可以单独进行训练，并且它们的预测能以某种方式结合起来去做出一个总体预测。

集成学习的基本思想**结合多个学习器组合成一个性能更好的学习器。**

该算法主要的问题是**要找出哪些较弱的模型可以结合起来，以及如何结合的方法**。

集成学习为什么有效？**不同的模型通常会在测试集上产生不同的误差；如果成员的误差是独立的，集成模型将显著地比其成员表现更好。**

##（一）组合方法

### 1 组合框架

| 序号 | 框架         | 基本思想 | 方法 | 特点 |      | 代表算法 |
| ---- | ------------ | -------- | ---- | ---- | ---- | -------- |
| 1    | Bagging      |          |      |      |      |          |
| 2    | Boosting     |          |      |      |      |          |
| 3    | Stacking[^2] |          |      |      |      |          |
|      |              |          |      |      |      |          |







## （二）Random Forest （RF)





## （三）GDBT







## （四） XGBoost



## （五）Adaboost

XGBoost(eXtreme Gradient Boosting)是Gradient Boosting算法的一个优化的版本。







# 注释：

[^1]:C: Classification; P: Prediction.
[^2]: 周志华-《机器学习》中没有将 Stacking 方法当作一种集成策略，而是作为一种结合策略，比如加权平均和投票都属于结合策略。