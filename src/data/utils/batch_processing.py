import random
from transformers import AutoTokenizer


class BatchProcessing:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       model_max_length=512)

    def tokenize(self, texts: list, padding="max_length", add_special_tokens=True):
        tokenized_text = self.tokenizer(texts, padding=padding,
                                        truncation='only_second',
                                        return_tensors="pt",
                                        add_special_tokens=add_special_tokens,
                                        return_token_type_ids=True)
        return tokenized_text

    def read_samples(self):
        pass

class TrainBatching(BatchProcessing):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def pos_neg_sampling(self, start, end, batch_size):
        idx = list(range(start, end))
        random.shuffle(idx)
        idx = idx[:batch_size // 2]
        return idx

    def build_training_batch(self, batch):
        return batch