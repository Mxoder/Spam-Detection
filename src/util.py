import chardet
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification


"""
编码检测
"""
def detect_encoding(
    file_path: str
) -> str:
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']



"""
加载 bert
"""
def load_bert(
    model_name_or_path: str
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    logger.info('Loading model...')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,          # 快速分词器
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
    )

    return tokenizer, model
