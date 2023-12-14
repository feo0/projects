import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from torchmetrics.functional import auroc
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import OneHotEncoder
from torchmetrics.functional import auroc
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
import re
from joblib import load
import pickle
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

#%matplotlib inline
#%config InlineBackend.figure_format='retina'
RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)

app = FastAPI()

# загрузка лейблов
with open('LABEL_COLUMNS.sav', 'rb') as f:
    LABEL_COLUMNS = pickle.load(f)

# ссылка на модель с huggingface
BERT_MODEL_NAME = 'cointegrated/rubert-tiny2'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

MAX_TOKEN_COUNT = 512

class CommentTagger(pl.LightningModule):
  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output
  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}
  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss
  def training_epoch_end(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)
    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    for i, name in enumerate(LABEL_COLUMNS):
      class_roc_auc = auroc(predictions[:, i], labels[:, i], task="binary", num_classes=len(LABEL_COLUMNS))
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=5e-4, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )

# Загрузка из контрольной точки
the_model = torch.load("best-checkpoint.ckpt")

model = CommentTagger(
  n_classes=len(LABEL_COLUMNS),
  n_warmup_steps=1,
  n_training_steps=1
)

model.load_state_dict(the_model['state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# заморозка всех слоев
model.eval()
model.freeze()

# загрузка параметров модели ovr из обучения
ovr = load('ovr.joblib')

pred_df = pd.DataFrame(columns=[["Текст","Предсказание"]])

class CommentsType(BaseModel):
    text: str

@app.post('/predict')
def predict(df_comments_pred : CommentsType):
    # предсказания по одному комментарию
    test_comment = str(df_comments_pred)
    # получим эмбеддинги и вероятности
    encoding = tokenizer.encode_plus(
      test_comment,
      add_special_tokens=True,
      max_length=512,
      return_token_type_ids=False,
      padding="max_length",
      return_attention_mask=True,
      return_tensors='pt',
    )
    _, test_prediction = model(encoding["input_ids"], encoding["attention_mask"])
    test_prediction = test_prediction.flatten().numpy()
    true_target_set = []
    # перевод в метки
    test_prediction_label = ovr.predict(test_prediction.reshape(1,len(LABEL_COLUMNS)))
    if test_prediction_label.sum() != 0:
        true_target_set = test_prediction_label
    else:
        max_val_1 = 0
        for i in test_prediction:
            if i > max_val_1:
                max_val_1 = i
            true_target = test_prediction > (max_val_1*0.75)
            true_false = []
            for true_1 in true_target:
                true_1 = int(true_1)
                true_false += [true_1]
        true_target_set += [true_false]
    labels_pred = []
    for i, z in zip(true_target_set[0], LABEL_COLUMNS):
        if i == 1:
            labels_pred += [z]
    labels_pred
    
    return {
    "prediction": str(labels_pred)[2:][:-2]
    }


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
