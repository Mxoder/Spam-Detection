# <center>机器学习大作业报告 - 垃圾邮件识别</center>





[TOC]

## 最终结果

![](resource\最终结果.png)

- 准确率：99.65%
- 排名：1



## 团队分工

- 王乐：前期调研、框架讨论、传统机器学习方法探究、SVM 实现
- 王艺杰：前期调研、框架讨论、数据分析、数据预处理与增强
- 张贤韬：前期调研、框架讨论、Naive Bayes 实现、Bert 实现、超参数调优探索
- 李泰川：前期调研、框架讨论、深度学习方法探究、LSTM 实现

<div style="page-break-before:always;"></div>

## 一、问题介绍



<div style="page-break-before:always;"></div>

## 二、方法描述

### （一）数据分析与预处理



### （二）模型探索

#### 1. Naive Bayes

一般有五种常用的朴素贝叶斯算法：

##### 1.1 BernoulliNB - 伯努利朴素贝叶斯
模型适用于多元伯努利分布，即每个特征都是二值变量，如果不是二值变量，该模型可以先对变量进行二值化。

在文档分类中特征是单词是否出现，如果该单词在某文件中出现了即为 1，否则为 0。

在文本分类中，统计词语是否出现的向量(word occurrence vectors)，而非统计词语出现次数的向量(word count vectors)。

BernoulliNB 可能在一些数据集上表现得更好，特别是那些更短的文档。如果时间允许，一般会对两个模型都进行评估。


##### 1.2 CategoricalNB - 类朴素贝叶斯
对分类分布的数据实施分类朴素贝叶斯算法，专用于离散数据集， 它假定由索引描述的每个特征都有其自己的分类分布。

对于训练集中的每个特征 $X$，CategoricalNB 估计以类 $y$ 为条件的 $X$ 的每个特征 $i$ 的分类分布。

样本的索引集定义为 $J = 1, …, m$，其中 $m$ 为样本数。


##### 1.3 GaussianNB - 高斯朴素贝叶斯
特征变量是连续变量，符合高斯分布，比如说人的身高，物体的长度。这种模型假设特征符合高斯分布。


##### 1.4 MultinomialNB - 多项式朴素贝叶斯
特征变量是离散变量，符合多项分布，在文档分类中特征变量体现在一个单词出现的次数，或者是单词的 TF-IDF 值等。

不支持负数，所以输入变量特征的时候，不用 `StandardScaler` 进行标准化数据，可以使用 `MinMaxScaler` 进行归一化。

这个模型假设特征复合多项式分布，是一种非常典型的文本分类模型，模型内部带有平滑参数 `alpha`。


##### 1.5 ComplementNB - 补充朴素贝叶斯
是 MultinomialNB 模型的一个变种，实现了补码朴素贝叶斯(CNB)算法。

CNB 是标准多项式朴素贝叶斯(MNB)算法的一种改进，比较适用于不平衡的数据集，在文本分类上的结果通常比 MultinomialNB 模型好。

具体来说，CNB 使用来自每个类的补数的统计数据来计算模型的权重。

CNB 的发明者的研究表明，CNB 的参数估计比 MNB 的参数估计更稳定。



#### 2. SVM



#### 3. LSTM



#### 4. BERT

BERT（Bidirectional Encoder Representations from Transformers）是一种基于 Transformer 架构的预训练语言模型。它通过无监督学习从大规模文本语料中预训练得到的通用语言表示，可以捕捉到丰富的语义信息。BERT 具有以下特点：

- 双向性：BERT 通过双向上下文建模来理解单词的含义，而不仅仅是依赖于左边或右边的上下文。
- 预训练和微调：BERT 首先在大规模语料上进行预训练，然后在特定任务上进行微调，因此可以适应各种自然语言处理任务。
- Transformer 架构：BERT 模型使用 Transformer 的编码器结构，使其能够处理长距离依赖关系，并具有较强的并行计算能力。
- 较强的语义理解能力：BERT 通过预训练学习到了丰富的语义信息，可以更好地理解文本含义

![](resource\bert原理结构图.png)

现如今，BERT 在自然语言处理领域中有广泛的应用，包括文本分类、命名实体识别、情感分析、问答系统等任务。它在这些任务中展现出了优异的性能，是当前最为广泛应用的模型之一。因此，我们重点考虑应用 BERT 来解决本问题。

我们还尝试了许多变体，其中最出色的是微软的 [DeBERTa](https://huggingface.co/microsoft/deberta-v3-base)，它有两大关键特性：

- 自注意力解耦机制：用 2 个向量分别表示 content 和 position，即 word 本身的文本内容和位置。word 之间的注意力权重则使用 word 内容之间和位置之间的解耦矩阵
- 增强的掩码解码器：传统解耦注意机制已经考虑了上下文单词的内容和**相对位置**，但没有考虑这些单词的**绝对位置**，DeBERTa 解决了这一问题

<div style="page-break-before:always;"></div>

## 三、具体训练

### （一）Naive Bayes



### （二）SVM



### （三）LSTM



### （四）BERT

#### 1. 超参数经验

##### （1）优化器

优化器我们最终采用 AdamW。AdamW 是对 Adam 算法的改进版本，它在 Adam 的基础上引入了权重衰减（weight decay）机制，用于控制参数的正则化。通过引入权重衰减，AdamW 可以更好地约束模型的复杂度，防止过拟合。

##### （2）学习率

学习率决定了每次参数更新的步长，是训练过程中的重要超参数。根据已有经验，对于 BERT 通常会选择较小的学习率（1e-5 ~ 5e-5）。较小的学习率可以使模型更稳定地收敛，并且能够避免梯度爆炸或梯度消失的问题。

##### （3）学习率变化策略

常见的学习率策略有 constant，cosine 等。constant 策略使用固定的学习率进行训练，适用于简单的任务和数据集。而 cosine 策略会在训练过程中逐渐降低学习率，在后期阶段，学习率会非常小，这有助于模型更好地收敛到最优解。在我们的尝试中，对于本任务采用 constant 策略即可。

##### （4）Loss

Loss 函数用于衡量模型预测结果与真实标签之间的差异，最经典的就是普通交叉熵。在数据分析阶段，我们知道了本任务的数据集是不均衡的，因此可以尝试不同的 Loss 函数来处理。

普通交叉熵损失函数适用于一般情况，它对所有样本平等对待。带 $α$ 的交叉熵损失函数可用于处理正负样本不均衡，通过调节 $α$ 参数来控制正负样本的重要性。而 Focal Loss 则可以更好地处理难易样本不均衡，它通过引入调制因子来增加难样本的权重，从而更关注难以分类的样本。

根据我们的试验，由于最终评判标准为准确率，一味增大难例判别其实会影响整体结果，最终采用普通的交叉熵，也取得了优异的结果。

##### （5）DropOut

Dropout是一种常用的正则化技术，在训练过程中随机丢弃一部分神经元，以减少过拟合。

对于大模型，如 BERT，较大的 Dropout 参数值（例如0.5）可以显著减少过拟合风险；而对于小模型，可以考虑减小或去除 Dropout 操作，以避免信息损失。

##### （6）训练参数

在微调 BERT 模型时，一般有全参训练或仅训练分类器头两种方式。全参训练是指对整个 BERT 模型进行微调，包括预训练得到的权重。这种方式通常适用于数据集较大、任务复杂的情况。而仅训练分类器头是指固定 BERT 模型的权重，只训练分类器层的参数。这种方式适用于数据集较小、任务相对简单的情况，可以减少计算资源的消耗。

根据我们的试验，最终采用了全参训练。

除了以上提到的超参数外，还有其他一些超参数也需要考虑。例如批量大小（batch size），它决定了每次迭代中训练的样本数量，需要根据显存大小和训练速度进行合理选择。训练轮数（epochs）指的是训练过程中完整遍历数据集的次数，过少的轮数可能导致欠拟合，过多的轮数可能导致过拟合。



#### 2. 自定义任务头

![](resource\分类器头.png)

我们定义一个分类器头，用于适配下游二分类任务。

```python
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()
```

其中 forward 如下：

```python
def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
```





#### 3. 技巧

##### （1）词元阈值过滤

##### （2）假标签法



<div style="page-break-before:always;"></div>

## 四、Optuna 自动调参

