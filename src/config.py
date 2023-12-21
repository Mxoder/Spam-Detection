import os


cur_dir = os.path.dirname(os.path.abspath(__file__))                    # project/src
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project

# data
data_dir = os.path.join(par_dir, 'data')                                # project/data
data_train_path = os.path.join(data_dir, 'train.csv')
data_test_path = os.path.join(data_dir, 'test.csv')

# model
model_dir = os.path.join(par_dir, 'model')                              # project/model
