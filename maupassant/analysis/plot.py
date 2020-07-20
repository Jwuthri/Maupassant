import matplotlib.pyplot as plt


def plot_elbow(distortion):
    """Show the explanation/distortion related to number of clusters"""
    if bool(len(distortion)):
        plt.plot(len(distortion), distortion, 'bx-')
    else:
        raise Exception("Please provide wcss")
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal n_clusters')
    plt.show()


def learning_curves(history, metric='macro_f1'):
    """Plot the learning curves of loss and macro f1 score"""
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_metric = history.history[metric]
    val_metric = history.history[f"val_{metric}"]

    return train_loss, val_loss, train_metric, val_metric
