import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(104, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        # x=x.view(-1,104)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def getDataLoader():
    df = pd.read_csv(r'../input/d/taboo00/adultdata/adult.data', header=None, encoding='utf-8')
    df.drop(columns=[2, 4, 10, 11], axis=1, inplace=True)
    df.columns = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'hours-per-week', 'native-country', 'salary']
    df.loc[df['salary'] == ' <=50K', 'salary'] = 0
    df.loc[df['salary'] == ' >50K', 'salary'] = 1
    df['age'] = df['age'].astype(int)
    df['hours-per-week'] = df['hours-per-week'].astype(int)
    df['salary'] = df['salary'].astype(int)
    df_categorical = [i for i in df.columns
                      if df[i].dtype.name == 'object']
    df_numerical = [i for i in df.columns
                    if df[i].dtype.name != 'object']

    df1 = pd.read_csv(r'../input/d/taboo00/adultdata/adult.test', header=None, encoding='utf-8')
    df1.drop(columns=[2, 4, 10, 11], axis=1, inplace=True)
    df1.columns = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'hours-per-week', 'native-country', 'salary']
    df1.loc[df['salary'] == ' <=50K.', 'salary'] = 0
    df1.loc[df['salary'] == ' >50K.', 'salary'] = 1
    df1['age'] = df['age'].astype(int)
    df1['hours-per-week'] = df['hours-per-week'].astype(int)
    df1['salary'] = df['salary'].astype(int)

    df2 = pd.concat([df, df1], axis=0)

    df_tree = pd.concat([df2[df_numerical],
                         pd.get_dummies(df2[df_categorical])], axis=1)
    df_x = df_tree.drop(['salary'], axis=1)
    df_y = df_tree['salary']

    train = torch.tensor(df_x[:32561].values)  # 将pandas转torch
    train = train.to(torch.float32)  # 将torch中的类型转化为float，因为有时pandas中格式不统一
    train_y = torch.tensor(df_y[:32561])  # 将pandas转torch
    train_y = train_y.to(torch.float32)  # 将torch中的类型转化为float，因为有时pandas中格式不统一
    # df_y = F.one_hot(df_y.long(), num_classes=2)   # n为类别数
    train_dataset = Data.TensorDataset(train, train_y)

    test = torch.tensor(df_x[32561:].values)  # 将pandas转torch
    test = test.to(torch.float32)  # 将torch中的类型转化为float，因为有时pandas中格式不统一
    test_y = torch.tensor(df_y[32561:])  # 将pandas转torch
    test_y = test_y.to(torch.float32)  # 将torch中的类型转化为float，因为有时pandas中格式不统一
    # df_y = F.one_hot(df_y.long(), num_classes=2)   # n为类别数
    test_dataset = Data.TensorDataset(test, test_y)
    return train_dataset, test_dataset


def train(torch_dataset, testData):
    trainloader = Data.DataLoader(dataset=torch_dataset,
                                  batch_size=128,  # x, y 是相差为1个数为10的等差数列, batch= 5, 遍历loader就只有两个数据
                                  shuffle=True,  # 不打乱顺序,便于查看
                                  num_workers=2)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.long().to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        test(net, testData)
        running_loss = 0.0
    return net


def test(net, torch_dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    testloader = Data.DataLoader(dataset=torch_dataset,
                                 batch_size=128,  # x, y 是相差为1个数为10的等差数列, batch= 5, 遍历loader就只有两个数据
                                 shuffle=False,  # 不打乱顺序,便于查看
                                 num_workers=2)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test : %d %%' % (100 * correct / total))


if __name__ == '__main__':
    trainData, testData = getDataLoader()
    net = train(trainData, testData)