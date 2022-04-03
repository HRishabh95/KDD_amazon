import os.path
import re
import os
import pandas as pd

data_path = '/Users/ricky/Documents/Rishabh/Dataset/KDD_amazon/'
prod_cat = pd.read_csv(f'''{data_path}product_cat.csv''')

## clean
def preprocessing(x):
    x = str(x).lower()
    x = re.sub(r'https?://\S+|www.\.\S+', '', x)
    x = re.sub(r'[^A-Za-zñáéíóúü0-9一-龠ぁ-ゔァ-ヴー々〆〤]+', ' ', x)
    return x



def index_file_for_each(cleaned_prod_cat,data_type):
    for lang in ['us','jp','es']:
        lang_prod = cleaned_prod_cat[cleaned_prod_cat['product_locale'] == lang]
        dupli = lang_prod.duplicated(subset='product_id', keep=False)
        lang_prod[dupli].sort_values('product_id')
        lang_prod_sub = lang_prod[
            ['product_id', data_type]]
        lang_prod[data_type] = lang_prod[data_type].apply(lambda x: preprocessing(x))
        lang_prod_sub.columns = ['docno', 'text']
        lang_prod_sub.to_csv('./subsets/%s_prod_%s.csv' %(lang,data_type), index=False)


def clean_df_for_each(prod_cat):
    subsets=['product_bullet_point', 'product_title', 'product_description', 'product_brand']
    for sub in subsets:
        cleaned_prod_cat = prod_cat.dropna(subset=[sub])
        ## duplicated and saving
        index_file_for_each(cleaned_prod_cat,sub)



clean_df_for_each(prod_cat)



def for_2_comb(lang_prod,data_type,lang):
    lang_prod_s = lang_prod[
        ['product_id', data_type[0], data_type[1]]]
    lang_prod_s[data_type[0]] = lang_prod_s[data_type[0]].apply(lambda x: preprocessing(x))
    lang_prod_s[data_type[1]] = lang_prod_s[data_type[1]].apply(lambda x: preprocessing(x))
    lang_prod_s[f'''{data_type[0]}_{data_type[1]}'''] = lang_prod_s[data_type[0]] + lang_prod_s[data_type[1]]
    lang_prod_sub = lang_prod_s[['product_id', f'''{data_type[0]}_{data_type[1]}''']]
    lang_prod_sub.columns = ['docno', 'text']
    lang_prod_sub.to_csv('./subsets/%s_prod_%s_%s.csv' % (lang, data_type[0], data_type[1]), index=False)

def for_3_comb(lang_prod,data_type,lang):
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

def for_4_comb(lang_prod,data_type,lang):
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
def index_file_for_combine(cleaned_prod_cat,data_type):
    for lang in ['us','jp','es']:
        print(lang,data_type)
        lang_prod = cleaned_prod_cat[cleaned_prod_cat['product_locale'] == lang]
        dupli = lang_prod.duplicated(subset='product_id', keep=False)
        lang_prod[dupli].sort_values('product_id')

        if len(data_type)==2:
            if not os.path.isfile('./subsets/%s_prod_%s_%s.csv' % (lang, data_type[0], data_type[1])):
                for_2_comb(lang_prod,data_type,lang)

        elif len(data_type)==3:
            if not os.path.isfile('./subsets/%s_prod_%s_%s_%s.csv' % (lang, data_type[0], data_type[1], data_type[2])):
                for_3_comb(lang_prod,data_type,lang)


        else:
            if not os.path.isfile('./subsets/%s_prod_%s_%s_%s_%s.csv' % (lang, data_type[0], data_type[1], data_type[2], data_type[3])):
                for_4_comb(lang_prod, data_type,lang)



def clean_df_combine(prod_cat):
    subsets = [['product_bullet_point', 'product_title'], ['product_bullet_point','product_description'],
               ['product_bullet_point','product_brand'],
               ['product_title','product_description'],
               ['product_brand','product_description','product_bullet_point'],
               ['product_title','product_description','product_bullet_point'],
               ['product_title', 'product_brand', 'product_bullet_point'],
               ['product_title','product_description','product_bullet_point','product_brand']]
    for sub in subsets:
        cleaned_prod_cat = prod_cat.dropna(subset=sub)
        ## duplicated and saving
        index_file_for_combine(cleaned_prod_cat, sub)


clean_df_combine(prod_cat)


## Read test
def test_topic(df, lang='en'):
    topics = df[['query_id', 'query']]
    topics = topics.drop_duplicates()
    topics.columns = [['qid', 'query']]

    for ii, rows in topics.iterrows():
        topics.at[ii, 'query'] = preprocessing(rows['query'])
    topics.to_csv('./subsets/%s_test_topics.csv' % lang, index=False, sep=':',header=False)
    return topics

data_path = '/Users/ricky/Documents/Rishabh/Dataset/KDD_amazon/'

test = pd.read_csv(f'''{data_path}test.csv''')
# use for each language.
for lang in ['us', 'es', 'jp']:
    test_sub = test[test["query_locale"] == lang]
    topics_test = test_topic(test_sub, lang=lang)