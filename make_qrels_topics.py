import os
import re

import pandas as pd


def mkdir_path(path):
    try:
        os.mkdirs(path)
    except:
        raise


def preprocessing(x):
    '''

    :param x: Df
    :return:
    '''
    x = str(x).lower()
    x = re.sub(r'https?://\S+|www.\.\S+', '', x)
    x = re.sub(r'[^A-Za-zñáéíóúü0-9一-龠ぁ-ゔァ-ヴー々〆〤]+', ' ', x)
    return x


def get_qrels_topics(df, lang):
    mkdir_path('./subsets')
    topics = df[['query_id', 'query']]
    topics = topics.drop_duplicates()
    topics.columns = [['qid', 'query']]

    for ii, rows in topics.iterrows():
        topics.at[ii, 'query'] = preprocessing(rows['query'])

    qrels = df[['query_id', 'product_id', 'esci_label']]
    qrels.columns = [['qid', 'docno', 'label']]

    qrels.replace({'exact': 3, 'substitute': 2, 'complement': 1, 'irrelevant': 0}, inplace=True)

    topics.to_csv('./subsets/%s_topics.csv' % lang, index=False, sep=':', header=False)
    qrels.to_csv('./subsets/%s_qrels.csv' % lang, index=False, sep=',')
    return qrels, topics


def test_topic(df, lang='en'):
    topics = df[['query_id', 'query']]
    topics = topics.drop_duplicates()
    topics.columns = [['qid', 'query']]

    for ii, rows in topics.iterrows():
        topics.at[ii, 'query'] = preprocessing(rows['query'])
    topics.to_csv('./subsets/%s_test_topics.csv' % lang, index=False, sep=':', header=False)
    return topics

