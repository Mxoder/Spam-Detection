import torch
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from util import detect_encoding


"""
直接读取数据
"""
def read_data_raw(
    file_path: str
) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding=detect_encoding(file_path))
    return df


"""
读取数据并整理
"""
def read_data_cleaned(
    file_path: str
) -> pd.DataFrame:
    df = read_data_raw(file_path)

    if 'train.csv' in file_path:
        # 将每行的第三列之后的内容合并到第三列
        df['v2'] = df.apply(lambda row: ','.join(map(str, filter(lambda x: pd.notna(x), row[2:]))), axis=1)
        df = df.iloc[:, :3]
        # 规范命名
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'content'})
    elif 'test.csv' in file_path:
        # 规范命名
        df = df[[0]].rename(columns={0: 'content'})
    else:
        logger.warning('Unsupported file, please check.')
    
    return df


"""
将训练集划分为内容与标签
"""
def split_content_label(
    df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    x = df['content']                                           # 内容
    y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)    # 标签
    return x, y


"""
自定义数据集（for bert）
"""
class BertDataset(Dataset):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.data = read_data_cleaned(file_path)

    def __getitem__(self, index):
        return self.data.iloc[index]['content'], self.data.iloc[index]['label']

    def __len__(self):
        return len(self.data)
