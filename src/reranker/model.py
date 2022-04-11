from abc import ABC

import numpy as np
import pytorch_lightning as pl
import transformers
from evaluator import Evaluator
from loss import PairwiseHingLoss
from torch import nn, split
from transformers import AdamW, AutoConfig, AutoModel


class CrossEncoder(pl.LightningModule, ABC):
    def __init__(self, model_name: str, num_labels: int = None, max_length: int = None,
                 n_training_steps=None, n_warmup_steps=None, actual_batch_size=16):
        super().__init__()
        self.n_training_steps = n_training_steps
        self.batch_size = actual_batch_size
        self.n_warmup_steps = n_warmup_steps

        """configuration"""
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        if num_labels is not None:
            self.config.num_labels = num_labels - 1
        """Modeling"""
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        try:
            self.out_size = self.bert.pooler.dense.out_features
        except:
            self.out_size = self.bert.config.dim
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.out_size, num_labels - 1)

        self.max_length = max_length

        self.criterion = PairwiseHingLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask).last_hidden_state

        linear_output = self.linear(
            self.dropout(sequence_output[:, 0, :].view(-1, self.out_size)))
        return linear_output

    def training_step(self, batch, batch_idx):
        model_predictions = self(batch["input_ids"],
                                 batch["attention_mask"],
                                 batch["token_type_ids"])

        model_predictions = self.sigmoid(model_predictions)
        model_predictions_p, model_predictions_n = split(model_predictions,
                                                         model_predictions.size(dim=0) // 2)

        loss_value = self.criterion(model_predictions_p[:, 0],
                                    model_predictions_n[:, 0])
        self.log("train_loss", loss_value, prog_bar=True, logger=True)
        return loss_value

    def eval_batch(self, batch):
        qids = batch["qid"]
        qids = qids if qids.__class__ == list else qids.tolist()
        qids = [str(i) for i in qids]
        labels = batch["label"]
        labels = labels if labels.__class__ == list else labels.tolist()
        scores = self(batch["input_ids"],
                      batch["attention_mask"],
                      batch["token_type_ids"])
        return {"qid": qids, "scores": scores, "labels": labels}

    def eval_epoch(self, outputs, name, epoch=-1):
        qids, labels, scores = [], [], []
        for output in outputs:
            qids.extend(output["qid"])
            labels.extend(output["labels"])
            scores.append(output["scores"].cpu().detach().numpy())

        scores = np.concatenate(scores, 0)
        evaluator = Evaluator.from_input_examples(qids=qids,
                                                  labels=labels,
                                                  scores=scores,
                                                  name=name,
                                                  metric='P_10')
        acc, metrics = evaluator(epoch=epoch)
        self.log("acc", acc, prog_bar=True, logger=True)
        [self.log(i[0], i[1], prog_bar=True, logger=True) for i in metrics]

    def validation_step(self, batch, batch_idx):
        return self.eval_batch(batch)

    def validation_epoch_end(self, outputs):
        self.eval_epoch(outputs, "during_training", self.current_epoch)

    def test_step(self, batch, batch_idx):
        return self.eval_batch(batch)

    def test_epoch_end(self, outputs):
        self.eval_epoch(outputs, "dev")

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        optimizer_class = AdamW
        optimizer_params = {'lr': 2e-5}
        linear = ['linear.weight', 'linear.bias']
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in linear, param_optimizer))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in linear, param_optimizer))))
        optimizer = optimizer_class([{'params': base_params}, {'params': params, 'lr': 1e-3}], **optimizer_params)

        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=self.n_warmup_steps,
                                                                 num_training_steps=self.n_training_steps)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )
