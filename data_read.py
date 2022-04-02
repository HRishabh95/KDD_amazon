import re

import pandas as pd

data_path = '/Users/ricky/Documents/Rishabh/Dataset/KDD_amazon/'
prod_cat = pd.read_csv(f'''{data_path}product_cat.csv''')

train = pd.read_csv(f'''{data_path}train.csv''')

## Missing values
cleaned_prod_cat = prod_cat.dropna(subset=['product_bullet_point', 'product_title', 'product_description'])

## duplicated
dupli = cleaned_prod_cat.duplicated(subset='product_id', keep=False)
cleaned_prod_cat[dupli].sort_values('product_id')

## language split
en_prod = cleaned_prod_cat[cleaned_prod_cat['product_locale'] == 'us']
jp_prod = cleaned_prod_cat[cleaned_prod_cat['product_locale'] == 'jp']
es_prod = cleaned_prod_cat[cleaned_prod_cat['product_locale'] == 'es']


## clean
def preprocessing(x):
    x = str(x).lower()
    x = re.sub(r'https?://\S+|www.\.\S+', '', x)
    x = re.sub(r'[^A-Za-zñáéíóúü0-9一-龠ぁ-ゔァ-ヴー々〆〤]+', ' ', x)
    return x


# Qrels
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

en_topics, en_qrels = get_qrels_topics(en_train, lang='en')
jp_topics, jp_qrels = get_qrels_topics(jp_train, lang='jp')
es_topics, es_qrels = get_qrels_topics(es_train, lang='es')


## get files to index in TREC format
def index_file(df_prod, lang='en'):
    df_prod.product_title = df_prod.product_title.apply(lambda x: preprocessing(x))
    df_prod.product_bullet_point = df_prod.product_bullet_point.apply(lambda x: preprocessing(x))
    df_prod.product_description = df_prod.product_description.apply(lambda x: preprocessing(x))
    en_prod_sub = df_prod[
        ['product_id', 'product_title', 'product_bullet_point', "product_brand", "product_description"]]
    en_prod_sub.columns = ['docno', 'title', 'text', 'brand', 'desc']
    en_prod_sub.brand = en_prod_sub.brand.str.lower()
    en_prod_sub['title_text'] = en_prod_sub['title'] + en_prod_sub['text']
    en_prod_sub['brand_text'] = en_prod_sub['brand'] + en_prod_sub['text']
    en_prod_sub['title_brand_text'] = en_prod_sub['title'] + en_prod_sub['brand'] + en_prod_sub['text']
    en_prod_sub['title_brand_des_text'] = en_prod_sub['title'] + en_prod_sub['brand'] + en_prod_sub['desc'] + \
                                          en_prod_sub['text']
    en_prod_sub['title_desc_text'] = en_prod_sub['title'] + en_prod_sub['desc'] + en_prod_sub['text']
    en_prod_sub.to_csv('./subsets/%s_prod_sub.csv' % lang, index=False)


# get files for each languge to index ('en','es' and 'jp')
for i in ['en', 'es', 'jp']:
    index_file(es_prod, lang=i)


## Read test
def test_topic(df, lang='en'):
    topics = df[['query_id', 'query']]
    topics = topics.drop_duplicates()
    topics.columns = [['qid', 'query']]

    for ii, rows in topics.iterrows():
        topics.at[ii, 'query'] = preprocessing(rows['query'])
    topics.to_csv('./subsets/%s_test_topics.csv' % lang, index=False, sep=':')
    return topics


test = pd.read_csv(f'''{data_path}test.csv''')
# use for each language.
for lang in ['en', 'es', 'jp']:
    test_sub = test[test["query_locale"] == lang]
    topics_test = test_topic(test_sub, lang=lang)
