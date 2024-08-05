import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    path = '/storage/mskim/thinfilm'
    train_data = pd.read_csv(path + '/csv/train.csv')
    # train_data = np.load(path + '/npy/train.npy')

    corr_matrix = train_data.corr()

    corr_matrix_1 = corr_matrix.loc[:4]

    # result_1 = []
    # result_2 = []
    # result_3 = []
    # result_4 = []
    # for i in train_data.columns[4:]:
    #     result_1.append(corr_matrix.loc['layer_1', i])
    #     result_2.append(corr_matrix.loc['layer_2', i])
    #     result_3.append(corr_matrix.loc['layer_3', i])
    #     result_4.append(corr_matrix.loc['layer_4', i])
    #
    # plt.figure(figsize=(15, 15))
    #
    # plt.plot(train_data.columns[4:], result_1, label='layer_1')
    # plt.plot(train_data.columns[4:], result_2, label='layer_2')
    # plt.plot(train_data.columns[4:], result_3, label='layer_3')
    # plt.plot(train_data.columns[4:], result_4, label='layer_4')

    # plt.plot(result_1, label='layer_1')
    # plt.plot(result_2, label='layer_2')
    # plt.plot(result_3, label='layer_3')
    # plt.plot(result_4, label='layer_4')

    # plt.title('Correlation between layer and wavelength')
    # plt.xlabel('wavelength')
    # plt.ylabel('correlation')
    # plt.legend()
    # plt.show()


    # plt.figure(figsize=(15, 15))
    # sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f")
    # plt.show()


    # plt.figure(figsize=(30, 30))
    # for y in y_data.columns:
    #     plt.plot(x_data, b[y], label=y)
    # plt.xlabel('layer_4')
    # plt.legend()
    # plt.show()
