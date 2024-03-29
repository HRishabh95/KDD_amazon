import re

import pandas as pd

data_path = '/Users/ricky/Documents/Rishabh/Dataset/KDD_amazon/'
prod_cat = pd.read_csv(f'''{data_path}product_cat.csv''')

train = pd.read_csv(f'''{data_path}train.csv''')

## Missing values
prod_cat = prod_cat.dropna(subset=['product_description','product_brand'])

## duplicated
dupli = prod_cat.duplicated(subset='product_id', keep=False)
prod_cat[dupli].sort_values('product_id')

## language split
en_prod = prod_cat[prod_cat['product_locale'] == 'us']
jp_prod = prod_cat[prod_cat['product_locale'] == 'jp']
es_prod = prod_cat[prod_cat['product_locale'] == 'es']

en_prod = en_prod[
        ['product_id', 'product_title', 'product_bullet_point', "product_brand", "product_description"]]
en_prod.columns = ['docno', 'title', 'text', 'brand', 'desc']

en_prod.to_csv('./subsets/us_prod_sub.csv', index=False)

es_prod = es_prod[
        ['product_id', 'product_title', 'product_bullet_point', "product_brand", "product_description"]]
es_prod.columns = ['docno', 'title', 'text', 'brand', 'desc']
es_prod.to_csv('./subsets/es_prod_sub.csv', index=False)

jp_prod = jp_prod[
        ['product_id', 'product_title', 'product_bullet_point', "product_brand", "product_description"]]
jp_prod.columns = ['docno', 'title', 'text', 'brand', 'desc']
jp_prod.to_csv('./subsets/jp_prod_sub.csv', index=False)


def preprocessing(x):
    x = str(x).lower()
    x = re.sub(r'https?://\S+|www.\.\S+', '', x)
    x = re.sub(r'[^A-Za-zñáéíóúü0-9一-龠ぁ-ゔァ-ヴー々〆〤]+', ' ', x)
    return x



## get files to index in TREC format for different combination
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
prod_sub={'en':en_prod,'es':es_prod,'jp':jp_prod}
for i in ['en', 'es', 'jp']:
    index_file(prod_sub[i], lang=i)