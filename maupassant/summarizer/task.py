import os
import click

import pandas as pd

from maupassant.summarizer.train import Trainer
from maupassant.summarizer.predict import Predictor


@click.command()
@click.argument("train_path", type=click.Path(exists=True))
@click.argument("test_path", type=click.Path(exists=True))
@click.argument("val_path", type=click.Path(exists=True))
@click.argument("feature", type=str)
@click.argument("label", type=str)
@click.option("architecture", default="CNN_NN", type=str)
@click.option("label_type", default="multi-label", type=str)
def train(train_path, test_path, val_path, feature, label, architecture, label_type):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    trainer = Trainer(train_df, test_df, val_df, label_type, architecture, feature, label)
    model_path = trainer.main()

    return model_path


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("example", type=str)
@click.argument("threshold", type=float)
def predict(model_path, example, threshold):
    path = os.path.join(model_path, "model")
    predictor = Predictor(path)
    pred = predictor.predict(example, threshold)
    print(pred)

    return pred
