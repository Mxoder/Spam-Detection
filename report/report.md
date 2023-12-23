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

我们五种方案均进行了探索，以 ComplementNB 为例、结合 Optuna 自动调参：

```python
import optuna

def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-10, 1.0)
    model = ComplementNB(alpha=alpha)
    model.fit(x_train_cnt, y_train)
    return model.score(x_test_cnt, y_test)

# 创建Optuna优化对象
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300)

# 输出最佳超参数取值
best_alpha = study.best_params['alpha']
best_test_value = study.best_value
print("Best alpha:", best_alpha)
print("Best test value: ", best_test_value)

# 用最佳超参数训练并预测
best_cnb = ComplementNB(alpha=best_alpha)
best_cnb.fit(x_train_cnt, y_train)
cnb_pred = best_mnb.predict(real_x_cnt)

# 输出
# ...
```

#### 2. SVM

#### 3. LSTM

长短时记忆网络（Long Short-Term Memory，LSTM）是一种递归神经网络（Recurrent Neural Network，RNN）的变体，用于处理序列数据。LSTM 的关键特点是能够捕捉长期依赖关系，避免了传统 RNN 在训练过程中遇到的梯度消失或梯度爆炸的问题。

##### 3.1 LSTM基本原理

LSTM 包含一个单元（cell），该单元中包含三个门（gates）：遗忘门（forget gate）、输入门（input gate）和输出门（output gate）。这三个门的作用如下：

- 遗忘门：决定是否丢弃过去的记忆。
- 输入门：决定更新当前时刻的记忆。
- 输出门：决定输出的记忆。

具体而言，对于每个时刻 t，LSTM 单元的状态（cell state）$C_t$ 和输出（output）$h_t$ 的计算如下：

1. 遗忘门：$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2. 输入门：$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$，$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
3. 更新记忆：$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$
4. 输出门：$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$，$h_t = o_t \cdot \tanh(C_t)$

其中，$[h_{t-1}, x_t]$ 表示将上一时刻的输出和当前时刻的输入连接起来，$W_f, b_f, W_i, b_i, W_C, b_C, W_o, b_o$ 是需要学习的参数，$\sigma$ 是 sigmoid 函数，$\tanh$ 是双曲正切函数。

##### 3.2 应用于垃圾短信识别的LSTM模型

在垃圾短信识别任务中，可以将每条短信看作一个序列，其中每个单词或字符作为序列的一个时间步。以下是本作业中垃圾短信分类的 LSTM 模型定义：

```python
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ...
```

上述模型包括一个嵌入层（Embedding）、一个LSTM层和一个全连接层。嵌入层将文本转换为密集向量表示，LSTM层用于捕捉序列信息，最后的全连接层输出二分类的结果。

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

#### 1. 无调优测试

定义一个 LSTM 分类模型（见**二、方法描述** -> **（二）模型探索** -> $3.2$），并使用训练集进行训练。在训练过程中，我们记录每个 epoch 的损失，并在验证集上评估模型的性能。

```python
# 创建数据加载器
train_dataset = LSTMDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
input_size = X_train_tensor.shape[1]
hidden_size = 128
output_size = 1
model = LSTMClassifier(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss = criterion(val_output.squeeze(), y_val_tensor)
        val_preds = (val_output.squeeze() >= 0.5).float()
        accuracy = torch.sum(val_preds == y_val_tensor).item() / len(y_val_tensor)

    print(f'Validation Loss: {val_loss.item()}, Accuracy: {accuracy}')
```

#### 2. 超参数调优

本模型较为简单，需要进行调优的超参数包括 **学习率** 和 **隐藏层神经元个数** 。在每个 epoch 结束后，我们记录损失，并根据这些信息搜索最佳的学习率和隐藏层神经元个数（具体搜索方式在**四、Optuna 自动调参**进行说明）：
```python
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

trial = study.best_trial
# 使用最佳超参数重新训练模型
best_hidden_size = study.best_params['hidden_size']
best_learning_rate = study.best_params['learning_rate']

# 初始化模型
best_model = LSTMClassifier(input_size, best_hidden_size, output_size)
best_optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)
best_criterion = nn.BCELoss()

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    best_model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        best_optimizer.zero_grad()
        output = best_model(batch_X)
        loss = best_criterion(output.squeeze(), batch_y)
        loss.backward()
        best_optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')
```

#### 3. 结果输出

最后，使用重新训练的模型在测试集上进行预测，并将结果输出到文件中。
```python
real_x = df_test['content']

# 在测试集上进行预测
real_x_vec = vectorizer.transform(real_x)
real_x_tensor = torch.tensor(real_x_vec.toarray(), dtype=torch.float32)

best_model.eval()
with torch.no_grad():
    test_output = best_model(real_x_tensor)
    test_preds = (test_output.squeeze() >= 0.5).float()

# 将预测结果转换为 DataFrame
df_result = pd.DataFrame({'content': real_x, 'prediction': test_preds.numpy()})

# 如果 prediction 列的值为 1，则代表 spam；否则，为 ham
df_result['label'] = df_result['prediction'].apply(lambda x: 'spam' if x == 1 else 'ham')
df_result = df_result[['content', 'label']]

# 将结果写入文件
with open('submission_lstm.txt', 'w', encoding='utf-8') as f:
    for res in df_result['label']:
        f.write(res + '\n')
```
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
        # pre_classifier 接收来自 bert 的最后一层隐藏状态作为输入，将其映射到更高维度的表示。
        # 这一层的目的是引入更多的非线性变换和学习能力，以更好地适应具体的分类任务。
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        # classifier 接收经过 pre_classifier 处理后的特征表示作为输入，将其映射到最终的输出类别数。
        # out_features 等于类别的数量，如二分类任务中 out_features 为 2。
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

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

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

首先，创建一个空的 set，用于存储只在 spam 中出现的 token。然后采用 BERT 的 tokenizer，遍历**训练集**中每行内容进行分词，统计**仅存在于 spam 中的 token**。

```python
# 创建一个空的set用于存储只在spam中出现的token
spam_only_tokens = set()

# 遍历每行内容进行分词
for index, row in df.iterrows():
    content = row["content"]
    label = row["label"]

    # 将内容进行分词
    tokens = tokenizer.tokenize(content)

    # 如果是spam，则将所有token添加到spam_only_tokens中
    if label == "spam":
        spam_only_tokens.update(tokens)
    elif label == "ham":
        # 如果是ham，则检查是否有在spam_only_tokens中的token，有的话从spam_only_tokens中移除
        for token in tokens:
            if token in spam_only_tokens:
                spam_only_tokens.remove(token)
```

然后采用四类过滤手段：

1. 垃圾词个数高于一定绝对数
2. 垃圾词占总长度比值高于一定百分比
3. 垃圾词个数低于一定绝对数
4. 垃圾词占总长度比值低于一定百分比

注意！这个过滤是**反向置信过滤**！

例如：采用第一种手段（垃圾词个数高于一定绝对数的就判定为 spam），那么**保证不了判定为 spam 的一定是 spam，只能保证不被判定为 spam 的一定为 ham**。通俗来说，就是这是一种**宁愿错杀、不肯放过**的手段，如果在这么严苛的情况下还没被判定为 spam，那么肯定就不是 spam 了。

用这种方法进行预过滤后再继续预测，可以提高一定表现。

##### （2）伪标签法（pseudo label）

伪标签主要思想也比较简单：

1. 在训练集上训练模型，并预测测试集的标签
2. 取测试集中预测置信度较高的样本（如预测为 1 的概率大于0.95），加入到训练集中
3. 使用新的训练集重新训练一个模型，并预测测试集的标签
4. 重复执行 2 和 3 步骤若干次（一至两次即可）

```python
# 准备已标记和未标记的数据集
labeled_dataset = ...  # 已标记数据集
unlabeled_dataset = ...  # 未标记数据集

# 创建数据加载器
labeled_dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)

# 初始化模型
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始训练，使用已标记数据
model.train()
for epoch in range(5):  # 初始训练若干轮
    for inputs, labels in labeled_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 使用伪标签进行训练
model.train()
for epoch in range(5):  # 使用伪标签训练若干轮
    pseudo_labels = []
    unlabeled_data = []
    for inputs, _ in unlabeled_dataloader:
        outputs = model(inputs)
        pseudo_labels.extend(torch.argmax(outputs, dim=1).tolist())
        unlabeled_data.append(inputs)

    # 将伪标签与未标记数据合并，形成新的标记数据集
    pseudo_labeled_dataset = torch.utils.data.TensorDataset(torch.cat(unlabeled_data, dim=0), torch.tensor(pseudo_labels))
    pseudo_labeled_dataloader = DataLoader(pseudo_labeled_dataset, batch_size=32, shuffle=True)

    for (inputs, labels), (pseudo_inputs, pseudo_labels) in zip(labeled_dataloader, pseudo_labeled_dataloader):
        optimizer.zero_grad()
        labeled_outputs = model(inputs)
        pseudo_labeled_outputs = model(pseudo_inputs)
        labeled_loss = criterion(labeled_outputs, labels)
        pseudo_labeled_loss = criterion(pseudo_labeled_outputs, pseudo_labels)
        loss = labeled_loss + pseudo_labeled_loss
        loss.backward()
        optimizer.step()
```

<div style="page-break-before:always;"></div>

## 四、Optuna 自动调参

Optuna 是一个强大的自动化超参数优化框架。它使用了一种称为“序列化和停止准则”的技术，通过迭代地评估不同的超参数组合来寻找最佳的模型性能。

相比于传统的 Grid Search 结合 k 折交叉验证，Optuna 性能更好、更高效，因此我们普遍采用了 Optuna 进行调参。

Optuna 的使用重点在于 `objective` 函数，定义待搜索的超参数空间，一个 Bert 的调参示例如下：

```python
# 定义目标函数，供 Optuna 调用
import torch.optim as optim
from transformers import AutoModelForSequenceClassification

logging_steps = 100

def objective(trial):
    # 定义超参数搜索空间
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True)
    eps = trial.suggest_float('eps', 1e-9, 1e-6, log=True)
    params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'eps': eps,
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
    )
    optimizer = optim.AdamW(model.parameters(), **params)

    global_step = 0
    model.to(device)

    model.train()
    for batch in tqdm(tr_loader):
        if device == 'cuda':
            batch = {k: v.cuda() for k, v in batch.items()}
        optimizer.zero_grad()
        output = model(**batch)
        output.loss.backward()
        optimizer.step()

        if (global_step + 1) % logging_steps == 0:
            print(f'\nsteps: {global_step}, loss: {output.loss.item()}', flush=True)
        global_step += 1

    # 进行评估
    acc = evaluate(model, val_ds, val_loader)
    print(f'\naccuracy: {acc}', flush=True)

    return acc  # Optuna 追求最大化目标
```
