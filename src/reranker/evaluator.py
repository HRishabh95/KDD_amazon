import os
import csv
from typing import List
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *

if not pt.started():
    pt.init()


class Evaluator:
    """
    This evaluator can be used with the CrossEncoder class.
    It is designed for CrossEncoders with 2 or more outputs. It measures
    P@20 of the predicted rank with the gold labels.
    """

    def __init__(self, sentence_pairs: List[List[str]] = None, labels: List[int] = None,
                 qids=None, scores=None,
                 name: str = '', write_csv: bool = True, metric="P_20"):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.qids = qids
        self.name = name
        self.metric = metric
        self.pred_scores = scores

        self.csv_file = "Evaluator" + ("_" + name if name else '') + "_results.csv"
        self.metrics = ['P_10', 'ndcg_cut_10', MAP@100, MRR@10]
        self.csv_headers = [str(i) for i in self.metrics]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List = None, **kwargs):

        if examples is not None:
            sentence_pairs = []
            labels = []

            for example in examples:
                sentence_pairs.append(example.texts)
                labels.append(example.label)
            return cls(sentence_pairs, labels, **kwargs)
        else:
            return cls(**kwargs)

    def __call__(self, model=None, output_path: str = "./", epoch: int = -1, steps: int = -1) -> float:
        if model is not None:
            pred_scores = model.predict(self.sentence_pairs,
                                        convert_to_numpy=True,
                                        show_progress_bar=False,
                                        apply_softmax=False)
            self.pred_scores = pred_scores

        df_scores = pd.DataFrame(
            {"qid": self.qids, "score": self.pred_scores[:, 0], "label": self.labels})
        df_scores.sort_values(by=["qid", "score"], ascending=False, inplace=True)

        # not required but helps to understand
        replacement_mapping_dict = {}
        for i, name in enumerate(df_scores["qid"].unique()):
            replacement_mapping_dict[name] = str(i)

        df_scores["qid"].replace(replacement_mapping_dict, inplace=True)

        df_scores["docno"] = df_scores.index
        df_scores["docno"] = df_scores["docno"].astype(str)

        res_columns = ['qid', 'docno', 'score']
        qrels_columns = ['qid', 'docno', 'label']

        if self.metric not in self.metrics:
            self.metrics += self.metric
            self.csv_headers = [str(i) for i in self.metrics]

        eval = pt.Utils.evaluate(df_scores[res_columns].copy(), df_scores[qrels_columns].copy(), self.metrics)
        acc = eval[self.metric]

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps] + [eval[str(metric)] for metric in self.metrics])

        return acc, [[str(metric), eval[str(metric)]] for metric in self.metrics]
