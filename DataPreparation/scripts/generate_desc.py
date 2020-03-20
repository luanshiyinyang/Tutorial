"""
Author: Zhou Chen
Date: 2020/3/20
Desc: desc
"""
import pandas as pd
import os
import glob


train_folder = '../data/Caltech101/train/'
valid_folder = '../data/Caltech101/valid/'
test_folder = '../data/Caltech101/test/'

train_desc = '../data/desc_train.csv'
valid_desc = '../data/desc_valid.csv'
test_desc = '../data/desc_test.csv'


def gen_csv(target_path, src_path):
    file_name = []
    label = []
    for category in os.listdir(src_path):
        for file in glob.glob(os.path.join(src_path, category) + '/*.jpg'):
            label.append(category)
            file_name.append(file.replace('\\', '/'))
    pd.DataFrame({'file_name': file_name, 'class': label}).to_csv(target_path, index=False, encoding='utf8')


if __name__ == '__main__':
    gen_csv(train_desc, train_folder)
    gen_csv(valid_desc, valid_folder)
    gen_csv(test_desc, test_folder)