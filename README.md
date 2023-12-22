# BUAA 2023 Machine Learning Course Final Project

本项目为 2023 BUAA 机器学习导论大作业——垃圾邮件识别任务。



## 一、任务简介

### 1. 任务内容

根据短信内容判断短信是否为垃圾短信。

- 数据规模：3537 条训练数据，2035 条测试数据。

- 输出格式：submission.zip - submission.txt

以测试集数据为顺序，每一行输出短信是否为垃圾短信。 

> spam：是垃圾短信；ham：不是垃圾短信。

### 2. 评分细则

评价方法：准确率（Accuracy）



## 二、方案内容

采用了 Naive Bayes，SVM，LSTM，Bert 四类方案，调参策略使用 Optuna 自动调参。

下面是项目文件夹说明：

```bash
.
|   LICENSE
|   README.md
|   requirements.txt	                # 项目依赖
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
|
+---report				# 技术报告存放路径
|   |   report.md
|   |   report.pptx		        # 展示报告
|   |
|   \---resource
|
\---src					# 源码文件夹
        config.py		        # 项目配置
        dataloader.py	                # 数据加载
        main.py			        # 主入口
        preprocess.py	                # 数据分析与预处理
        train.py		        # 训练
        util.py			        # 其他工具
```



## 三、使用说明

### 1. 环境搭建

> 推荐使用 virtualenv 或 conda 配置虚拟环境

在项目文件夹下，执行命令：

```bash
pip3 install -r requirements.txt
```

### 2. 具体使用

> 不想写太杂，没有支持命令行调用

在 src/main.py 中，选择模型，执行代码即可。
