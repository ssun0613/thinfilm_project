import csv
import pandas as pd

def train(path):
    split_data = pd.read_csv(path + '/split/layer_1(10).csv')
    a = split_data[split_data['layer_3'] == 10]
    a = a.iloc[:, 1:]
    new_index = range(1, 1 + len(a))
    a = a.set_index(pd.Index(new_index))

    new_df_10 = a
    for i in range(1, 901, 5):
        new_df_10 = new_df_10.drop(i)

    for k in range(20, 310, 10):

        a = split_data[split_data['layer_3'] == k]
        a = a.iloc[:, 1:]
        new_index = range(1, 1 + len(a))
        a = a.set_index(pd.Index(new_index))

        new_df = a
        for i in range(1, 901, 5):
            new_df = new_df.drop(i)

        new_df_10 = pd.concat([new_df_10, new_df], ignore_index=True)

    result = new_df_10

    for j in range(20, 310, 10):
        split_data = pd.read_csv(path + '/split/layer_1({}).csv'.format(j))
        a = split_data[split_data['layer_3'] == 10]
        a = a.iloc[:, 1:]
        new_index = range(1, 1 + len(a))
        a = a.set_index(pd.Index(new_index))

        new_df_10 = a
        for i in range(1, 901, 5):
            new_df_10 = new_df_10.drop(i)

        for k in range(20, 310, 10):

            a = split_data[split_data['layer_3'] == k]
            a = a.iloc[:, 1:]
            new_index = range(1, 1 + len(a))
            a = a.set_index(pd.Index(new_index))

            new_df = a
            for i in range(1, 901, 5):
                new_df = new_df.drop(i)

            new_df_10 = pd.concat([new_df_10, new_df], ignore_index=True)

        result = pd.concat([result, new_df_10], ignore_index=True)

    result.to_csv(path + 'train_ssun.csv', sep=',')

def test(path):
    ###### test_ssun.csv

    split_data = pd.read_csv(path + '/split/layer_1(10).csv')
    a = split_data[split_data['layer_3'] == 10]
    a = a.iloc[:, 1:]
    new_index = range(1, 1 + len(a))
    a = a.set_index(pd.Index(new_index))

    new_df_10 = pd.DataFrame(a.iloc[0:1])
    for i in range(5, 901, 5):
        new_df_10 = pd.concat([new_df_10, a.iloc[i:i + 1]], ignore_index=True)

    for k in range(20, 310, 10):

        a = split_data[split_data['layer_3'] == k]
        a = a.iloc[:, 1:]
        new_index = range(1, 1 + len(a))
        a = a.set_index(pd.Index(new_index))

        new_df = pd.DataFrame(a.iloc[0:1])
        for i in range(5, 901, 5):
            new_df = pd.concat([new_df, a.iloc[i:i + 1]], ignore_index=True)

        new_df_10 = pd.concat([new_df_10, new_df], ignore_index=True)

    result = new_df_10

    for j in range(20, 310, 10):
        split_data = pd.read_csv(path + '/split/layer_1({}).csv'.format(j))
        a = split_data[split_data['layer_3'] == 10]
        a = a.iloc[:, 1:]
        new_index = range(1, 1 + len(a))
        a = a.set_index(pd.Index(new_index))

        new_df_10 = pd.DataFrame(a.iloc[0:1])
        for i in range(5, 901, 5):
            new_df_10 = pd.concat([new_df_10, a.iloc[i:i + 1]], ignore_index=True)

        for k in range(20, 310, 10):

            a = split_data[split_data['layer_3'] == k]
            a = a.iloc[:, 1:]
            new_index = range(1, 1 + len(a))
            a = a.set_index(pd.Index(new_index))

            new_df = pd.DataFrame(a.iloc[0:1])
            for i in range(5, 901, 5):
                new_df = pd.concat([new_df, a.iloc[i:i + 1]], ignore_index=True)

            new_df_10 = pd.concat([new_df_10, new_df], ignore_index=True)

        result = pd.concat([result, new_df_10], ignore_index=True)

    result.to_csv(path + 'test_ssun.csv', sep=',')

def layer_avg(path):
    train_data = pd.read_csv(path + '/csv/train.csv')
    col = 'layer_4'
    layer = train_data[train_data[col] == 10]

    for i in layer[layer.columns.difference([col])].columns[-3:]:
        layer = layer.drop(columns=i)
    layer_s1 = layer.describe().iloc[1, :]

    layer_summary = pd.DataFrame(layer_s1).transpose()

    for k in range(20, 310, 10):
        layer = train_data[train_data[col] == k]

        for i in layer[layer.columns.difference([col])].columns[-3:]:
            layer = layer.drop(columns=i)
        layer_s2 = pd.DataFrame(layer.describe().iloc[1, :]).transpose()

        layer_summary = pd.concat([layer_summary, layer_s2], ignore_index=True)

    layer_summary.to_csv(path + '/csv/' + col + '.csv')

if __name__ == '__main__':
    path = '/storage/mskim/thinfilm'
    # train_data = pd.read_csv(path + '/train.csv')
    # split_data = pd.read_csv(path + '/split/layer_1(10).csv')

    # for i in range(10, 310, 10):
    #     train_data[train_data['layer_1'] == i].to_csv(path + '/split/layer_1({}).csv'.format(i), sep=',')

    # train(path)
    # test(path)
    # layer_avg(path)





