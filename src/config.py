import os


cur_dir = os.path.dirname(os.path.abspath(__file__))                    # project/src
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project

#### data ####
"""
数据路径
"""
data_dir = os.path.join(par_dir, 'data')                                # project/data
data_train_path = os.path.join(data_dir, 'train.csv')
data_test_path = os.path.join(data_dir, 'test.csv')


#### model ####
"""
模型路径
如果是预训练模型，默认从 huggingface 加载，或者设置为本地路径
这里全部设置为远程加载了
"""
model_dir = os.path.join(par_dir, 'model')                              # project/model
bert_path = 'bert-base-uncased'
distilbert_path = 'distilbert-base-uncased'
albert_path = 'albert-base-v2'
roberta_path = 'roberta-large'
deberta_path = 'microsoft/deberta-v3-base'


#### submission ####
submission_dir = os.path.join(par_dir, 'submission')
submission_path = os.path.join(submission_dir, 'submission.txt')
