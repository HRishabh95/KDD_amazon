import pyterrier as pt
import os
import sys
import pandas as pd

import pytrec_eval
if not pt.started():
    pt.init(mem=8000, version='snapshot',
            boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"]
           )

lang='es'
data_path='/Users/ricky/PycharmProjects/KDD_amazon/subsets/'

df_docs_es=pd.read_csv(f'''{data_path}es_prod_sub.csv''')
df_docs_en=pd.read_csv(f'''{data_path}en_prod_sub.csv''',index_col=0)

df_docs=pd.concat([df_docs_en,df_docs_es])
df_docs_drop=df_docs.drop_duplicates(subset = 'docno')

index_path='/Users/ricky/Documents/Rishabh/Dataset/KDD_amazon/index/'
data_type=f'''en_es_title_text'''
if not os.path.exists(f'''{index_path}{data_type}/data.properties'''):
    indexer = pt.DFIndexer(f'''{index_path}{data_type}''', overwrite=True, verbose=True, Threads=8)
    indexref3 = indexer.index(df_docs_drop["title_text"], df_docs_drop[['docno','title_text']])
else:
    indexref3 = pt.IndexRef.of(f'''{index_path}{data_type}/data.properties''')

BM25 = pt.BatchRetrieve(indexref3, num_results=30,  wmodel="BM25", controls={"c" : 0.8, "bm25.k_1": 0.6, "bm25.k_3": 0.5}, properties={
    'tokeniser': 'UTFTokeniser',
    'termpipelines': 'Stopwords',})

df_topics_es=pt.io.read_topics(f'''{data_path}es_topics.csv''',format='singleline',tokenise=True)
df_topics_en=pt.io.read_topics(f'''{data_path}en_topics.csv''',format='singleline',tokenise=True)

df_topics=pd.concat([df_topics_en,df_topics_es])
df_topics_drop=df_topics.drop_duplicates(subset = 'qid')

df_qrels_es=pd.read_csv(f'''{data_path}es_qrels.csv''',sep=',')
df_qrels_en=pd.read_csv(f'''{data_path}en_qrels.csv''',sep=',')

df_qrels=pd.concat([df_qrels_en,df_qrels_es])
df_qrels['label']=df_qrels['label'].astype(int)
df_qrels['qid']=df_qrels['qid'].astype(str)

res=BM25.transform(df_topics_drop)
print(pt.Utils.evaluate(res,df_qrels,metrics=['ndcg_cut_10','ndcg_cut_20','ndcg']))
