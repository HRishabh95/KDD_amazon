import random
from transformers import AutoTokenizer
from os.path import join as joinpath
import json
import pandas as pd


class BatchProcessing:
    def __init__(self, model_name: str,
                 query_rels_fn: str,
                 data_fn: str,
                 topics_fn: str,
                 query_rels_path: str = '../../../data/interim',
                 data_path: str = '../../../data/processed'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       model_max_length=512)

        self.query_rels = json.loads(open(joinpath(query_rels_path, query_rels_fn), 'r').read())
        self.data = pd.read_csv(joinpath(data_path, data_fn))
        self.topics = pd.read_csv(joinpath(topics_fn, topics_fn))

    def tokenize(self, texts: list, padding="max_length", add_special_tokens=True):
        tokenized_text = self.tokenizer(texts, padding=padding,
                                        truncation='only_second',
                                        return_tensors="pt",
                                        add_special_tokens=add_special_tokens,
                                        return_token_type_ids=True)
        return tokenized_text

    def assign_text_2_examples(self, batch: list):
        batch = [
            [self.topics.loc[self.topics.qid == i[0]]['query'].values[0],
             self.data.loc[self.data.qid == i[1]]['doc'].values[0]]
            for i in batch
        ]

        return batch


class TrainBatching(BatchProcessing):
    def __init__(self, model_name, batch_size, query_rels_fn, data_fn, topics_fn):
        super().__init__(model_name, query_rels_fn, data_fn, topics_fn)
        self.batch_size = batch_size

    def build_batch(self, batch):
        """
        :param batch: list of query ids candidates for training step
        :return: batch of tokenized examples for training

        examples need to be sample from the BM25 retrieved results
        positive examples are those present in both qrels and BM25
        negative samples are those 0-labeled or non-labeled and bottom BM25
        """
        pos_examples, neg_examples = [], []
        for qid in self.query_rels.keys:
            if qid in batch:
                pos_examples.append([
                    [qid, docno]
                    for docno in self.query_rels[qid]['BM25'] if docno in self.query_rels[qid]['qrels']
                ])
                neg_examples.append([
                                        [qid, docno]
                                        for docno in self.query_rels[qid]['BM25'] if
                                        docno not in self.query_rels[qid]['qrels']
                                    ][-50:])

        random.shuffle(pos_examples), random.shuffle(neg_examples)
        examples = pos_examples[:self.batch_size // 2] + neg_examples[:self.batch_size // 2]

        batch = self.assign_text_2_examples(examples)
        batch = self.tokenize(batch)

        return batch


class EvalBatching(BatchProcessing):
    def __init__(self, model_name, query_rels_fn, data_fn, topics_fn):
        super().__init__(model_name, query_rels_fn, data_fn, topics_fn)

    def get_examples(self, items, truncate_val=None):
        examples = []
        for qid in self.query_rels.keys:
            if qid in items:
                rank = self.query_rels[qid]['BM25']
                if truncate_val:
                    rank = rank[truncate_val]
                for docno in rank:
                    label = 1 if docno in self.query_rels[qid]['qrels'] else 0
                    examples.append([qid, docno, label])

        return examples

    def build_batch(self, batch):
        """
        :param batch: list of pairs with labels
        :return: batch of tokenized examples for training
        """
        examples = [[i[0], i[0]] for i in batch]
        labels = [i[1] for i in batch]
        batch = self.assign_text_2_examples(examples)
        batch = self.tokenize(batch)

        return batch, labels
