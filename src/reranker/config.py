import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.KDD_amazonDataModule import KDD_amazonDataModule
from model import CrossEncoder
import random
import yaml


with open('./config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)["reranker"]


random.seed(config["SEED"])
BATCH_SIZE = config["BATCH_SIZE"]
N_EPOCHS = config["N_EPOCHS"]


train_examples = train_examples[:]
valid_examples = valid_examples[:]
dev_examples = dev_examples[:]

model = CrossEncoder(
    model_name=config["model_name"],
    num_labels=2,
    max_length=512,
    n_warmup_steps=config["WARMUP_STEPS"],
    n_training_steps=len(train_examples) / BATCH_SIZE * N_EPOCHS,
    actual_batch_size=config["BATCH_SIZE"],
)

data_module = KDD_amazonDataModule(
    train_data=train_examples,
    val_data=valid_examples,
    test_data=dev_examples,
    train_batch_size=train_batch_size,
    batch_size=BATCH_SIZE,
)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint_{}".format(i),
    save_top_k=1,
    verbose=True,
    monitor="acc",
    mode="max"
)

logger = TensorBoardLogger("./reranker_logs", name=config["logger_name"])
early_stopping_callback = EarlyStopping(monitor='acc', patience=config["PATIENCE"], mode='max')

trainer = pl.Trainer(
    logger=logger,
    callbacks=[early_stopping_callback, checkpoint_callback],
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=50,
    accumulate_grad_batches=config["ACCUM_ITER"],
    check_val_every_n_epoch=config["EVAL_EVERY_N_EPOCH"]
)
trainer.fit(model, data_module)

trainer.test(dataloaders=data_module.test_dataloader())
