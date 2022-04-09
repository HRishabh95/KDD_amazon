import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.KDD_amazonDataModule import KDD_amazonDataModule
from model import CrossEncoder
import random
import yaml
from dotmap import DotMap
import json
from data.utils.batch_processing import EvalBatching as EB


with open('../../config.yml') as f:
    g_config = yaml.load(f, Loader=yaml.FullLoader)
    config = g_config["reranker"]
    data_config = g_config["dataset"]
    del g_config
    config = DotMap(config)
    data_config = DotMap(data_config)

train, val, test = [
    json.loads(open(config.data_path + i.format(config.language)).read())
    for i in ['train_{}.json', 'val_{}.json', 'test_{}.json']
                   ]

EB_val = EB(config.model_name, config.val_rels, data_config.collection, data_config.train)
val = EB_val.get_examples(val, truncate_val=100)
del EB_val
EB_test = EB(config.model_name, config.test_rels, data_config.collection, data_config.test)
test = EB_val.get_examples(test)
del EB_test

random.seed(config.seed)
BATCH_SIZE = config.batch_size
N_EPOCHS = config.n_epochs

expected_batches = config.n_examples // BATCH_SIZE
train_batch_size = len(train) // expected_batches

model = CrossEncoder(
    model_name=config.model_name,
    num_labels=config.n_labels,
    max_length=config.tk_max_len,
    n_warmup_steps=config.warmup_steps,
    n_training_steps=expected_batches * BATCH_SIZE * N_EPOCHS,
    actual_batch_size=BATCH_SIZE,
)

data_module = KDD_amazonDataModule(
    train_data=train,
    val_data=val,
    test_data=test,
    train_batch_size=train_batch_size,
    batch_size=BATCH_SIZE,
    model_name=config.model,
    train_rels=config.train_rels,
    val_rels=config.val_rels,
    test_rels=config.test_rels,
    data_fn=data_config.collection,
    topics_fn=data_config.train,
    test_topics_fn=data_config.test
)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint_{}".format(config.language),
    save_top_k=1,
    verbose=True,
    monitor="acc",
    mode="max"
)

logger = TensorBoardLogger("./reranker_logs", name=config.logger_name + config.language)
early_stopping_callback = EarlyStopping(monitor='acc', patience=config.patience, mode='max')

trainer = pl.Trainer(
    logger=logger,
    callbacks=[early_stopping_callback, checkpoint_callback],
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=50,
    accumulate_grad_batches=config.accum_iter,
    check_val_every_n_epoch=config.eval_every_n_epoch
)

trainer.fit(model, data_module)

trainer.test(dataloaders=data_module.test_dataloader())
