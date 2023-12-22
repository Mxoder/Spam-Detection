import torch
import optuna
import optuna.logging
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from config import *
from util import *
from dataloader import *


# 设置 Optuna 的日志级别为 WARNING，即 WARNING 以上才输出
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
Naive Bayes
"""
def naive_bayes_train() -> ComplementNB:
    # Optuna 调参
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-10, 1.0)
        model = ComplementNB(alpha=alpha)
        model.fit(x_train_cnt, y_train)
        return model.score(x_test_cnt, y_test)
    
    # 读取数据
    df_train = read_data_cleaned(data_train_path)
    df_test = read_data_cleaned(data_test_path)

    x, y = split_content_label(df_train)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_real = df_test['content']

    # 将文本数据转换为词袋模型的表示形式，即将文本转换为词频矩阵
    cv = CountVectorizer()
    x_train_cnt = cv.fit_transform(x_train.values)
    x_test_cnt = cv.transform(x_test.values)
    x_real_cnt = cv.transform(x_real.values)

    # 创建 Optuna 优化对象
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=300)

    # 输出最佳超参数取值
    best_alpha = study.best_params['alpha']
    best_test_value = study.best_value
    print("Best alpha:", best_alpha)
    print("Best test value: ", best_test_value)

    # 用最佳超参数训练并预测
    best_mnb = ComplementNB(alpha=best_alpha)
    best_mnb.fit(x_train_cnt, y_train)
    pred = best_mnb.predict(x_real_cnt)


"""
SVM
"""




"""
LSTM
"""





"""
Bert
"""
# 评估函数
def evaluate(model, dataset, loader):
    model.eval()
    acc_num = 0

    with torch.inference_mode():
        for batch in tqdm(loader):
            if device == 'cuda':
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()

    return acc_num / len(dataset)


# 训练函数
def train(model,
          optimizer,
          tr_loader,
          val_ds,
          val_loader,
          num_epochs=5,
          logging_steps=50):
    global_step = 0
    model.to(device)

    for epoch in range(num_epochs):
        print(f"current epoch: {epoch + 1} {'=' * 80}", flush=True)
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

        # 每个 epoch 后都进行评估
        acc = evaluate(model, val_ds, val_loader)
        print(f'\naccuracy: {acc}', flush=True)


# 推断函数（没有用批推理）
def inference(content, tokenizer, model):
    with torch.inference_mode():
        inputs = tokenizer(content, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1)
        return 'spam' if pred.item() == 1 else 'ham'


def bert_train():
    # Collator
    def collate_fn(batch):
        contents, labels = [], []
        for item in batch:
            contents.append(item[0])
            labels.append(1 if item[1] == 'spam' else 0)
        # max_length 是前面的 0.99 分位点
        inputs = tokenizer(contents,
                           max_length=288,
                           padding='max_length',
                           truncation=True,
                           return_tensors='pt')
        inputs['labels'] = torch.tensor(labels)
        return inputs
    

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
            distilbert_path,
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

            if (global_step + 1) % 50 == 0:
                print(f'\nsteps: {global_step}, loss: {output.loss.item()}', flush=True)
            global_step += 1

        # 进行评估
        acc = evaluate(model, val_ds, val_loader)
        print(f'\naccuracy: {acc}', flush=True)

        return acc  # Optuna 追求最大化目标
    

    # 读取数据并划分训练集，验证集
    ds = BertDataset(data_train_path)
    tr_ds, val_ds = random_split(ds, lengths=[0.8, 0.2])    # train_dataset, valid_dataset

    # 加载模型
    tokenizer, model = load_bert(distilbert_path)

    # DataLoader
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # 优化器选用 AdamW
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Optuna 自动调参
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=50)

    # 获得最佳超参数，可用于训练
    # best_params = study.best_params
    # best_accuracy = study.best_value
    # print("Best Hyperparameters:", best_params)
    # print("Best Accuracy:", best_accuracy)

    # 普通训练
    train(model, optimizer, tr_loader, val_ds, val_loader)

    # 输出结果
    df_test = read_data_cleaned(data_test_path)
    pred_list = [inference(c, tokenizer, model) for c in tqdm(df_test['content'])]

    with open(submission_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(pred_list))
