import json
import random

import pandas as pd
import yaml


def build_samples_4_reranking(
        bm25_res: pd.DataFrame,
        qrels: pd.DataFrame,
        language: str = 'en',
        mode: str = 'val'):

    # take bm25 results and relevant documents
    query_rels = {}
    queries = qrels.qid.astype(str).unique().tolist()
    for qid in queries:
        query_rels[qid] = {
            'qrels': bm25_res[bm25_res.qid == qid].docno.tolist(),
            'bm25': qrels[qrels.qid == qid].docno.tolist()
        }

    if mode == 'val':
        with open('./config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['dataset']
        random.seed(config['seed'])
        random.shuffle(queries)
        n_train = len(queries) * config['train_split_rate'] // 100
        splits = [queries[:n_train], queries[n_train:]]
        f_names = ['query_rels', 'train', 'val']
    else:
        splits = [queries]
        f_names = ['test_query_rels', 'test']

    out_file = "./data/interim/{0}_{1}.json".format("{}", language)
    for f_name, object_ in zip(f_names, [query_rels] + splits):
        open(out_file.format(f_name), "w").write(json.dumps(object_, indent=4))




