import matplotlib.pyplot as plt


def plot_elbow(distortion):
    if bool(len(distortion)):
        plt.plot(len(distortion), distortion, 'bx-')
    else:
        raise Exception("Please provide wcss")
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal n_clusters')
    plt.show()