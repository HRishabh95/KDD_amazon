import os
import os.path
import re
import yaml
import pandas as pd
from os.path import join as joinpath
from make_qrels_topics import get_qrels_topics,test_topic



## clean
def preprocessing(x):
    x = str(x).lower()
    x = re.sub(r'https?://\S+|www.\.\S+', '', x)
    x = re.sub(r'[^A-Za-zñáéíóúü0-9一-龠ぁ-ゔァ-ヴー々〆〤]+', ' ', x)
    return x


def index_file_for_each(cleaned_prod_cat, data_to_index):
    '''

    :param cleaned_prod_cat: Cleaned Product category Df
    :param data_to_index: Which text to index
    :return:
    '''
    for lang in ['us', 'jp', 'es']:
        lang_prod = cleaned_prod_cat[cleaned_prod_cat['product_locale'] == lang]
        dupli = lang_prod.duplicated(subset='product_id', keep=False)
        lang_prod[dupli].sort_values('product_id')
        lang_prod_sub = lang_prod[
            ['product_id', data_to_index]]
        lang_prod[data_to_index] = lang_prod[data_to_index].apply(lambda x: preprocessing(x))
        lang_prod_sub.columns = ['docno', 'text']
        lang_prod_sub.to_csv('./subsets/%s_prod_%s.csv' % (lang, data_to_index), index=False)


def clean_df_for_each(prod_cat):
    subsets = ['product_bullet_point', 'product_title', 'product_description', 'product_brand']
    for sub in subsets:
        cleaned_prod_cat = prod_cat.dropna(subset=[sub])
        ## duplicated and saving
        index_file_for_each(cleaned_prod_cat, sub)



def for_2_comb(lang_prod, data_type, lang):
    '''

    For 2 combination of text
    '''
    lang_prod_s = lang_prod[
        ['product_id', data_type[0], data_type[1]]]
    lang_prod_s[data_type[0]] = lang_prod_s[data_type[0]].apply(lambda x: preprocessing(x))
    lang_prod_s[data_type[1]] = lang_prod_s[data_type[1]].apply(lambda x: preprocessing(x))
    lang_prod_s[f'''{data_type[0]}_{data_type[1]}'''] = lang_prod_s[data_type[0]] + lang_prod_s[data_type[1]]
    lang_prod_sub = lang_prod_s[['product_id', f'''{data_type[0]}_{data_type[1]}''']]
    lang_prod_sub.columns = ['docno', 'text']
    lang_prod_sub.to_csv('./subsets/%s_prod_%s_%s.csv' % (lang, data_type[0], data_type[1]), index=False)


def for_3_comb(lang_prod, data_type, lang):
    '''

    For 3 combination of text
    '''
    lang_prod_s = lang_prod[
        ['product_id', data_type[0], data_type[1], data_type[2]]]
    lang_prod_s[data_type[0]] = lang_prod_s[data_type[0]].apply(lambda x: preprocessing(x))
    lang_prod_s[data_type[1]] = lang_prod_s[data_type[1]].apply(lambda x: preprocessing(x))
    lang_prod_s[data_type[2]] = lang_prod_s[data_type[2]].apply(lambda x: preprocessing(x))
    lang_prod_s[f'''{data_type[0]}_{data_type[1]}_{data_type[2]}'''] = lang_prod_s[data_type[0]] + lang_prod_s[
        data_type[1]] + lang_prod_s[data_type[2]]
    lang_prod_sub = lang_prod_s[['product_id', f'''{data_type[0]}_{data_type[1]}_{data_type[2]}''']]
    lang_prod_sub.columns = ['docno', 'text']
    lang_prod_sub.to_csv('./subsets/%s_prod_%s_%s_%s.csv' % (lang, data_type[0], data_type[1], data_type[2]),
                         index=False)


def for_4_comb(lang_prod, data_type, lang):
    '''

    For 4 combination of text
    '''
    lang_prod_s = lang_prod[
        ['product_id', data_type[0], data_type[1], data_type[2], data_type[3]]]
    lang_prod_s[data_type[0]] = lang_prod_s[data_type[0]].apply(lambda x: preprocessing(x))
    lang_prod_s[data_type[1]] = lang_prod_s[data_type[1]].apply(lambda x: preprocessing(x))
    lang_prod_s[data_type[2]] = lang_prod_s[data_type[2]].apply(lambda x: preprocessing(x))
    lang_prod_s[data_type[3]] = lang_prod_s[data_type[3]].apply(lambda x: preprocessing(x))

    lang_prod_s[f'''{data_type[0]}_{data_type[1]}_{data_type[2]}_{data_type[3]}'''] = \
        lang_prod_s[data_type[0]] + lang_prod_s[data_type[1]] + lang_prod_s[data_type[2]] + lang_prod_s[data_type[3]]

    lang_prod_sub = lang_prod_s[['product_id', f'''{data_type[0]}_{data_type[1]}_{data_type[2]}_{data_type[3]}''']]
    lang_prod_sub.columns = ['docno', 'text']
    lang_prod_sub.to_csv(
        './subsets/%s_prod_%s_%s_%s_%s.csv' % (lang, data_type[0], data_type[1], data_type[2], data_type[3]),
        index=False)


### Different combinations
def index_file_for_combine(cleaned_prod_cat, data_type):
    '''
    Create index file for combinations

    '''
    for lang in ['us', 'jp', 'es']:
        print(lang, data_type)
        lang_prod = cleaned_prod_cat[cleaned_prod_cat['product_locale'] == lang]
        dupli = lang_prod.duplicated(subset='product_id', keep=False)
        lang_prod[dupli].sort_values('product_id')

        if len(data_type) == 2:
            if not os.path.isfile('./subsets/%s_prod_%s_%s.csv' % (lang, data_type[0], data_type[1])):
                for_2_comb(lang_prod, data_type, lang)

        elif len(data_type) == 3:
            if not os.path.isfile('./subsets/%s_prod_%s_%s_%s.csv' % (lang, data_type[0], data_type[1], data_type[2])):
                for_3_comb(lang_prod, data_type, lang)


        else:
            if not os.path.isfile('./subsets/%s_prod_%s_%s_%s_%s.csv' % (
            lang, data_type[0], data_type[1], data_type[2], data_type[3])):
                for_4_comb(lang_prod, data_type, lang)


def clean_df_combine(prod_cat):
    '''

    Different combinations
    '''
    subsets = [['product_bullet_point', 'product_title'], ['product_bullet_point', 'product_description'],
               ['product_bullet_point', 'product_brand'],
               ['product_title', 'product_description'],
               ['product_brand', 'product_description', 'product_bullet_point'],
               ['product_title', 'product_description', 'product_bullet_point'],
               ['product_title', 'product_brand', 'product_bullet_point'],
               ['product_title', 'product_description', 'product_bullet_point', 'product_brand']]
    for sub in subsets:
        cleaned_prod_cat = prod_cat.dropna(subset=sub)
        ## duplicated and saving
        index_file_for_combine(cleaned_prod_cat, sub)



if __name__ == '__main__':
    with open('./config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['dataset']

    data_path = config['path']

    prod_cat = pd.read_csv(joinpath(data_path, config['collection']))
    train = pd.read_csv(joinpath(data_path, config['train']))

    # Get Topics and Qrels
    en_train = train[train['query_locale'] == 'us']
    jp_train = train[train['query_locale'] == 'jp']
    es_train = train[train['query_locale'] == 'es']

    en_topics, en_qrels = get_qrels_topics(en_train, lang='us')
    jp_topics, jp_qrels = get_qrels_topics(jp_train, lang='jp')
    es_topics, es_qrels = get_qrels_topics(es_train, lang='es')

    # index for each text fields such as title, product brand etc.
    if config['index_each_data']:
        clean_df_for_each(prod_cat)

    # index for combination of text fields.
    clean_df_combine(prod_cat)

    test = pd.read_csv(joinpath(data_path, config['test']))
    # use for each language.
    for lang in ['us', 'es', 'jp']:
        test_sub = test[test["query_locale"] == lang]
        topics_test = test_topic(test_sub, lang=lang)
