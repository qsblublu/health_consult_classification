import pandas as pd
import jieba
import re
import os
import json
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


with open('../../data/stopwords.txt', mode='r') as f:
    stop_words = [line.strip() for line in f.readlines()]


word2vec_data = []
disease_analysis_class = ['中医', '手术', '点滴', '休息', '多喝水吃水果蔬菜']


def is_stop_words(word):
    for stop_word in stop_words:
        if word == stop_word:
            return True
    return False


def remove_stop_words(seg_list):
    removed_list = []

    for word in seg_list:
        if not is_stop_words(word):
            removed_list.append(word)

    word2vec_data.append(removed_list)
    return removed_list


def clean_data(col):
    col = re.sub(r'病情分析|指导意见', '', col)
    col = re.sub(r'[.,();，。（）\-【】：！？/～；+*=‘’\']', '', col)
    col = re.sub(r' +', '', col)
    return col


def parse_column(column):
    pd.set_option('mode.chained_assignment', None)
    # parse column
    print(f'--------------------- start parse column {column.name} ------------------------')

    for idx, value in tqdm(column.items(), total=len(column)):
        if pd.isna(value):
            continue

        segment_list = jieba.lcut(clean_data(value))
        if len(segment_list) > 0:
            column[idx] = ' '.join(remove_stop_words(segment_list))
        else:
            column[idx] = np.nan

    print(f'--------------------- end parse column {column.name} ------------------------')


def cal_similarity(model, sentence, cls):
    try:
        words = sentence.split(' ')
        similarity = [model.wv.similarity(word, cls) for word in words]
        return np.max(similarity)
    except:
        return -1


def build_word2vec_model():
    word2vec_data.append(disease_analysis_class)

    # train word2vec model
    word2vec_model = Word2Vec(word2vec_data, min_count=1)
    word2vec_model.save('../../model/disease_analysis_word2vec.model')
    print('save word2vec model to ../../model/disease_analysis_word2vec.model')


def clean_null_of_df(df: pd.DataFrame):
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


def cal_label(df: pd.DataFrame, column):
    word2vec_model = Word2Vec.load('../../model/disease_analysis_word2vec.model')
    clean_null_of_df(df)

    sentence_cls = []
    error_index = []
    column = df[column]
    print('------------------ start cal disease analysis label ------------------------')

    for idx, sentence in tqdm(column.items(), total=len(column)):
        sentence_cls_similarity = []
        for cls in disease_analysis_class:
            sentence_cls_similarity.append(cal_similarity(word2vec_model, sentence, cls))

        if -1 in sentence_cls_similarity:
            error_index.append(idx)
        else:
            sentence_cls.append(disease_analysis_class[sentence_cls_similarity.index(max(sentence_cls_similarity))])

    df.drop(error_index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['disease_analysis_class'] = sentence_cls

    print('------------------ end cal disease analysis label ------------------------')


def build_dict(df: pd.DataFrame, column):
    word_dict = {}
    word_num = 0
    max_sentence_len = 0
    error_index = []
    column = df[column]

    print('----------------- start build word dict for health consult --------------------')
    for idx, sentence in tqdm(column.items(), total=len(column)):
        try:
            words = sentence.split(' ')
            max_sentence_len = max(max_sentence_len, len(words))
            for word in words:
                if word not in word_dict.keys():
                    word_num += 1
                    word_dict[word] = word_num
        except:
            error_index.append(idx)

    df.drop(error_index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f'health consult words total num is {word_num} max sentence length is {max_sentence_len}')

    word_dict['max_sentence_len'] = max_sentence_len

    with open('../../data/expr2/health_consult_dict.json', mode='w') as f:
        json.dump(word_dict, f)

    print('save word dict to ../../data/expr2/health_consult_dict.json')
    print(f'----------------- end build word dict for health consult ------------------------')


def build_data(all_data: pd.DataFrame):
    print(f'-------------------- start build data for health consult -----------------------------')
    clean_null_of_df(all_data)
    all_data.to_csv('../../data/expr2/health_consult_data.csv', index=False)
    print('save health consult data to ./../data/expr2/health_consult_data.csv')

    # save train and test data
    health_consult_data_index = all_data.index.to_numpy()
    train_index, test_index = train_test_split(health_consult_data_index, test_size=0.2, random_state=0)
    health_consult_data_train = all_data.iloc[train_index]
    health_consult_data_test = all_data.iloc[test_index]

    health_consult_data_train.to_csv('../../data/expr2/health_consult_data_train.csv', index=False)
    health_consult_data_test.to_csv('../../data/expr2/health_consult_data_test.csv', index=False)

    print('save health consult train data to ../../expr2/data/health_consult_data_train.csv')
    print('save health consult test data to ../../data/expr2/health_consult_data_test.csv')
    print(f'------------------- end build data for health consult -----------------------------')


def main():
    all_data = pd.read_csv('../../data/all_data.csv')
    all_data = all_data[['ID', '性别', '岁数', '健康咨询描述', '病情分析']]

    parse_column(all_data['健康咨询描述'])
    parse_column(all_data['病情分析'])

    build_word2vec_model()

    cal_label(all_data, '病情分析')
    build_dict(all_data, '健康咨询描述')

    all_data.to_csv('tmp.csv', index=False)
    all_data = pd.read_csv('tmp.csv')

    build_data(all_data)
    os.remove('tmp.csv')


if __name__ == '__main__':
    main()


