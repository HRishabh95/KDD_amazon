import re

import pandas as pd

data_path = '/Users/ricky/Documents/Rishabh/Dataset/KDD_amazon/'

train = pd.read_csv(f'''{data_path}train.csv''')

def preprocessing(x):
    x = str(x).lower()
    x = re.sub(r'https?://\S+|www.\.\S+', '', x)
    x = re.sub(r'[^A-Za-zñáéíóúü0-9一-龠ぁ-ゔァ-ヴー々〆〤]+', ' ', x)
    return x


def get_qrels_topics(df, lang):
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


# Get Topics and Qrels
en_train = train[train['query_locale'] == 'us']
jp_train = train[train['query_locale'] == 'jp']
es_train = train[train['query_locale'] == 'es']

en_topics, en_qrels = get_qrels_topics(en_train, lang='us')
jp_topics, jp_qrels = get_qrels_topics(jp_train, lang='jp')
es_topics, es_qrels = get_qrels_topics(es_train, lang='es')