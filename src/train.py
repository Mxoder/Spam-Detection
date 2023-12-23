import torch
import optuna
import optuna.logging
from torch import nn
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


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))  # 添加一个维度
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


def lstm_train():
    # 读取数据
    df_train = read_data_cleaned(data_train_path)
    df_test = read_data_cleaned(data_test_path)

    x, y = split_content_label(df_train)
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    # 使用CountVectorizer将文本转换为词袋模型
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # 转换为PyTorch Tensor
    X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.map({'ham': 0, 'spam': 1}).values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_vec.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.map({'ham': 0, 'spam': 1}).values, dtype=torch.float32)

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

    def objective(trial):
        # 定义超参数搜索空间
        hidden_size = trial.suggest_int('hidden_size', 32, 256)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        # 初始化模型
        model = LSTMClassifier(input_size, hidden_size, output_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 在每个epoch结束后记录损失
            trial.report(total_loss, epoch)

            # 判断是否应该提前停止训练
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 返回验证集上的准确度作为目标值
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output.squeeze(), y_val_tensor)
            val_preds = (val_output.squeeze() >= 0.5).float()
            accuracy = torch.sum(val_preds == y_val_tensor).item() / len(y_val_tensor)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # 打印最佳超参数和目标值
    print('Best trial:')
    trial = study.best_trial

    print('Value: {}'.format(trial.value))
    print('Params: ')
    for key, value in trial.params.items():
        print('{}: {}'.format(key, value))

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

    with open('submission_lstm.txt', 'w', encoding='utf-8') as f:
        for res in df_result['label']:
            f.write(res + '\n')


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
    tr_ds, val_ds = random_split(ds, lengths=[0.8, 0.2])  # train_dataset, valid_dataset

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
