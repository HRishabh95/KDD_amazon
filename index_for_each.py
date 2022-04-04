import os

import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init(mem=8000,version='snapshot', boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

def combine_result(data_path, model_name, data_to_index, dtype='train'):
    '''

    Combine results from different indexers.

    :param data_path: Data path
    :param model_name: BM25
    :param data_to_index: Combination to index
    :param dtype: train or test
    :return:
    '''
    final = []
    for lang in ['us', 'es', 'jp']:
        if os.path.isfile(f'''{data_path}{lang}_{data_to_index}_{model_name}_{dtype}.csv'''):
            sub_df = pd.read_csv(f'''{data_path}{lang}_{data_to_index}_{model_name}_{dtype}.csv''', index_col=0)
            if dtype == 'train':
                sub_df=sub_df[['qid','docno','score']]
                sub_df.qid=sub_df.qid.astype(str)
                sub_df.docno = sub_df.docno.astype(str)
                #sub_df.columns=['query_id','product_id','score']
            final.append(sub_df)


    df = pd.concat(final)
    if dtype=='test':
        df.query_id = df.query_id.astype(str)
        df.product_id=df.product_id.astype(str)
    df.to_csv(f'''{data_to_index}_{model_name}_{dtype}.csv''', index=None)
    return df


def check_the_submission(df,data_path,model_name,data_to_index):
    '''
    Check the submission files

    :param df: Combined Df
    :param data_path: Data path
    :param model_name: BM25
    :param data_to_index: Combination
    :return:
    '''
    cleaned_df=[]
    for lang in ['us','es','jp']:
        df_docs = pd.read_csv(f'''{data_path}{lang}_prod_sub.csv''')
        product_id=df_docs.docno.values
        product_id_combined = df[df['product_locale'] == lang].product_id.values
        common_product_id = list(set(product_id).intersection(product_id_combined))
        cleaned_df.append(df[(df['product_id'].isin(common_product_id)) & (df['product_locale'] == lang)])

    final_df=pd.concat(cleaned_df)
    final_df=final_df[['query_id','product_id']]
    final_df.to_csv(f'''{data_to_index}_{model_name}_test.csv''', index=None)
    return final_df

def get_merged_qrels(data_path):
    final=[]
    for lang in ['us','es','jp']:
        if os.path.isfile(f'''{data_path}{lang}_qrels.csv'''):
            qrels = pd.read_csv(f'''{data_path}{lang}_qrels.csv''', sep=',')
            qrels['label'] = qrels['label'].astype(int)
            qrels['qid'] = qrels['qid'].astype(str)
            final.append(qrels)
    df = pd.concat(final)
    return df

def get_qrels(data_path, lang):
    '''

    :param data_path: Data path for qrels specific to language (use make_qrels_topics.py before)
    :param lang: 'us','es','jp'
    :return:
    '''
    qrels = pd.read_csv(f'''{data_path}{lang}_qrels.csv''', sep=',')
    qrels['label'] = qrels['label'].astype(int)
    qrels['qid'] = qrels['qid'].astype(str)
    return qrels


def index_each_lang(qe=False, eval=True, lang='en', index_path='', gs=False, data_to_index='title_text',
                    change_settings=False,num_results=100):

    '''
    Index each language

    :param qe: Query Expansion
    :param eval: Evaluatiton and scoring on train
    :param lang: Specific Language ('us','es','jp')
    :param index_path: Path to save index
    :param gs: Grid Search
    :param data_to_index: Combination of Data
    :param change_settings: To use Grid Search settings
    :param num_results: Total number of results.
    :return:
    '''

    data_type = f'''{lang}_{data_to_index}'''
    df_docs = pd.read_csv(f'''./subsets/{lang}_prod_{data_to_index}.csv''')
    if not os.path.exists(f'''{index_path}{data_type}/data.properties'''):
        indexer = pt.DFIndexer(f'''{index_path}{data_type}''', overwrite=True, verbose=True, Threads=8)
        indexer.setProperty("tokeniser",
                            "UTFTokeniser")  # Replaces the default EnglishTokeniser, which makes assumptions specific to English
        indexer.setProperty("termpipelines", "Stopwords")
        indexref = indexer.index(df_docs['text'], df_docs[['docno', 'text']])
    else:
        indexref = pt.IndexRef.of(f'''{index_path}{data_type}/data.properties''')

    print('-----------------Indexed----------------\n')

    if lang=='us':
        stemmer='Stopwords,PorterStemmer'
    elif lang=='es':
        stemmer='Stopwords,SpanishSnowballStemmer'
    else:
        stemmer=''
    BM25 = pt.BatchRetrieve(indexref, num_results=num_results, wmodel="BM25",
                            controls={"c": 0.8, "bm25.k_1": 0.6, "bm25.k_3": 0.5}, properties={
            'tokeniser': 'UTFTokeniser',
            'termpipelines': stemmer, })
    model_name = 'BM25'

    if gs:
        print('--------------------Grid search--------------\n')
        qrels = get_qrels(data_path, lang)
        topics = pt.io.read_topics(f'''{data_path}{lang}_topics.csv''', format='singleline', tokenise=True)
        f = pt.GridSearch(
            BM25,
            {BM25: {"c": [0.3, 0.5, 0.8, 0.9],
                    "bm25.k_1": [0.3, 0.6, 0.9,1.2,1.4,1.6],
                    "bm25.k_3": [0.5, 2, 4, 6,8]
                    }},
            topics,
            qrels,
            "ndcg_cut_10", verbose=True)
        print(f)
        if change_settings:
            return True

    if qe:
        print('------------Query Expansion-------------')
        RM3 = pt.rewrite.RM3(indexref)
        BM25 = BM25 >> RM3 >> BM25
        model_name = 'BM25_RM3'

    if eval:
        print('-----------Evaluation--------------------')
        qrels = get_qrels(data_path, lang)
        topics = pt.io.read_topics(f'''{data_path}{lang}_topics.csv''', format='singleline', tokenise=True)
        res = BM25.transform(topics)
        res.to_csv(f'''{data_path}{lang}_{data_to_index}_{model_name}_train.csv''')
        print(pt.Utils.evaluate(res, qrels, metrics=['ndcg_cut_10', 'ndcg_cut_20', 'ndcg']))

    topics = pt.io.read_topics(f'''{data_path}{lang}_test_topics.csv''', format='singleline', tokenise=True)
    res = BM25.transform(topics)

    submission_sample = res[['qid', 'docno']]
    submission_sample.columns = ['query_id', 'product_id']
    submission_sample['product_locale'] = lang
    submission_sample.to_csv(f'''{data_path}{lang}_{data_to_index}_{model_name}_test.csv''')


data_path = '/Users/ricky/PycharmProjects/KDD_amazon/subsets/'
index_path = '/Users/ricky/Documents/Rishabh/Dataset/KDD_amazon/index/'

data_to_indexs=['product_bullet_point_product_title', 'product_bullet_point_product_description',
               'product_bullet_point_product_brand',
               'product_title_product_description',
               'product_brand_product_description_product_bullet_point',
               'product_title_product_description_product_bullet_point',
               'product_title_product_brand_product_bullet_point',
               'product_title_product_description_product_bullet_point_product_brand']
data_to_index = data_to_indexs[-2]


for lang in ['us','es','jp']:
    index_each_lang(qe=False, eval=True, lang=lang, index_path=index_path, data_to_index=data_to_index, gs=False,num_results=100)




create_submission_file=True
## Combining different results for submission.
if create_submission_file:
    combined_df=combine_result(data_path, 'BM25', data_to_index, dtype='test')
    cleaned_df=check_the_submission(combined_df,data_path,'BM25',data_to_index)




# train dtype
evaluate_train=False
if evaluate_train:
    combined_df=combine_result(data_path, 'BM25', data_to_index, dtype='train')
    qrels=get_merged_qrels(data_path)
    pt.Utils.evaluate(combined_df, qrels, metrics=['ndcg_cut_10', 'ndcg_cut_20'])
#{'ndcg_cut_10': 0.23747557699680494, 'ndcg_cut_20': 0.22943625462979889} num_results=20
#{'ndcg_cut_10': 0.27361675817105907, 'ndcg_cut_20': 0.26223407618626343} num_results=100