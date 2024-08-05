import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


if __name__ == '__main__':
    path = '/storage/mskim/thinfilm/csv/'
    # layer_1 = pd.read_csv(path + 'layer_1.csv')
    layer_2 = pd.read_csv(path + 'layer_2.csv')
    # layer_3 = pd.read_csv(path + 'layer_3.csv')
    # layer_4 = pd.read_csv(path + 'layer_4.csv')

    X = layer_2['layer_2']
    Y = range(len(layer_2.columns) - 1)
    Z = layer_2.iloc[:,1:].values

    Y, X = np.meshgrid(Y, X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z)

    # plt.title('layer_1')
    plt.show()


