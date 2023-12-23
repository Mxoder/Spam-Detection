from loguru import logger

from config import *
from preprocess import preprocess
from train import naive_bayes_train, lstm_train
from train import bert_train


# 朴素贝叶斯
def naive_bayes():
    logger.info('Model: Naive Bayes')
    naive_bayes_train()


# 支持向量机
def svm():
    logger.info('Model: SVM')
    # todo


# LSTM
def lstm():
    logger.info('Model: LSTM')
    lstm_train()
    # todo


# Bert
def bert():
    logger.info('Model: Distilbert-base-uncased')
    bert_train()


def main():
    # 数据分析 + 预处理
    logger.info('Preprocessing data...')
    preprocess()

    # 调用不同的模型训练、预测
    logger.info('Training model...')
    # naive_bayes()
    # svm()
    lstm()
    bert()

    logger.info(f'Training completed. The predictions have been saved to {submission_path}.')


if __name__ == '__main__':
    main()
