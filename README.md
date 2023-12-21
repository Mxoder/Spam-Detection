# BUAA 2023 Machine Learning Course Final Project

​		本项目为 2023 BUAA 机器学习课程大作业——垃圾邮件识别任务。



## 一、任务简介

### 1. 任务内容

​		根据短信内容判断短信是否为垃圾短信。

- 数据规模：3537 条训练数据，2035 条测试数据。

- 输出格式：submission.zip - submission.txt

​		以测试集数据为顺序，每一行输出短信是否为垃圾短信。 

> spam：是垃圾短信；ham：不是垃圾短信。

### 2. 评分细则

​		评价方法：准确率（Accuracy）



## 二、仓库内容

```bash
.
|   LICENSE
|   README.md
|
+---data
|       test.csv			# 测试集（预测目标）
|       train.csv			# 训练集
|
+---model				# 模型存放路径
|       __init__.py
|
+---notebook				# jupyter notebook 存放路径
|       spam_ham.ipynb
|       spam_ham_2.ipynb
|       spam_ham_3.ipynb
|
+---report				# 技术报告存放路径
|   |   report.md
|   |
|   \---resource
|
\---src					# 源码文件夹
```





## 三、使用说明

### 1. 环境搭建

> 推荐使用 virtualenv 或 conda 配置虚拟环境

​		在项目文件夹下，执行命令：

```bash
pip3 install -r requirements.txt
```

### 2. 具体使用

