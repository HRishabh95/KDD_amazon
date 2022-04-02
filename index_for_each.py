import os

import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init(mem=8000, version='snapshot',
            boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"]
            )


def combine_result(data_path, model_name, data_to_index, dtype='train'):
    final = []
    for lang in ['en', 'es', 'jp']:
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

def get_merged_qrels(data_path):
    final=[]
    for lang in ['en','es','jp']:
        if os.path.isfile(f'''{data_path}{lang}_qrels.csv'''):
            qrels = pd.read_csv(f'''{data_path}{lang}_qrels.csv''', sep=',')
            qrels['label'] = qrels['label'].astype(int)
            qrels['qid'] = qrels['qid'].astype(str)
            final.append(qrels)
    df = pd.concat(final)
    return df

def get_qrels(data_path, lang):
    qrels = pd.read_csv(f'''{data_path}{lang}_qrels.csv''', sep=',')
    qrels['label'] = qrels['label'].astype(int)
    qrels['qid'] = qrels['qid'].astype(str)
    return qrels


def index_each_lang(qe=False, eval=True, lang='en', index_path='', gs=False, data_to_index='title_text',
                    change_settings=False):

    data_type = f'''{lang}_{data_to_index}'''
    df_docs = pd.read_csv(f'''{data_path}{lang}_prod_sub.csv''')
    if not os.path.exists(f'''{index_path}{data_type}/data.properties'''):
        indexer = pt.DFIndexer(f'''{index_path}{data_type}''', overwrite=True, verbose=True, Threads=8)
        indexref3 = indexer.index(df_docs[data_to_index], df_docs[['docno', data_to_index]])
    else:
        indexref3 = pt.IndexRef.of(f'''{index_path}{data_type}/data.properties''')

    print('-----------------Indexed----------------\n')

    if lang=='en':
        stemmer='Stopwords,PorterStemmer'
    elif lang=='es':
        stemmer='Stopwords,SpanishSnowballStemmer'
    else:
        stemmer=''
    BM25 = pt.BatchRetrieve(indexref3, num_results=30, wmodel="BM25",
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
            {BM25: {"c": [0, 0.5, 0.8],
                    "bm25.k_1": [0.4, 0.6, 0.9, 1.2],
                    "bm25.k_3": [0.5, 2, 4, 6]
                    }},
            topics,
            qrels,
            "ndcg_cut_10", verbose=True)
        print(f)
        if change_settings:
            return True

    if qe:
        print('------------Query Expansion-------------')
        RM3 = pt.rewrite.RM3(indexref3)
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

    submission_sample.to_csv(f'''{data_path}{lang}_{data_to_index}_{model_name}_test.csv''')


data_path = '/Users/ricky/PycharmProjects/KDD_amazon/subsets/'
index_path = '/Users/ricky/Documents/Rishabh/Dataset/KDD_amazon/index/'
data_to_index = 'title_text'
for lang in ['es','jp']:
    index_each_lang(qe=False, eval=True, lang=lang, index_path=index_path, gs=False)
combined_df=combine_result(data_path, 'BM25', data_to_index, dtype='test')
qrels=get_merged_qrels(data_path)
pt.Utils.evaluate(combined_df, qrels, metrics=['ndcg_cut_10', 'ndcg_cut_20'])
#{'ndcg_cut_10': 0.23747557699680494, 'ndcg_cut_20': 0.22943625462979889}