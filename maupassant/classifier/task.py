import click

from maupassant.classifier.train import TrainClassifier
from maupassant.classifier.predict import Predictor


@click.command()
@click.option("method", type=click.Choice(['train', 'predict']))
@click.option("dataset_dir", type=str)
@click.option("features", default="", type=list)
@click.option("labels", default="", type=dict)
def run(method, dataset_dir, features, labels):
    """Run a specific task, as training or predicting."""
    if method == "train":
        trainer = TrainClassifier(dataset_dir, features, labels).fit_model()
    else:
        predicter = Predictor(dataset_dir)
