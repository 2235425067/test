import numpy as np
import torch
import os
import torch.utils.data as Data
import cv2
from matplotlib import pyplot as plt

List=[]
lableList=[]
lable=['an2i','at33','boland','bpm','ch4f','cheyer','choon','danieln','glickman','karyadi'
    ,'kawamura','kk49','megak','mitchell','night','phoebe','saavik','steffi','sz24','tammo']
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=30,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(192,100)
        self.mlp2 = torch.nn.Linear(100,20)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x

def getId(string):
    for i in range(len(lable)):
        if(lable[i] in string) :
            return i
    print("bug  ",string)
    return 0
def readFile(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    for file in files:  # 遍历文件夹
        file = os.path.join(path, file)

        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            img_cv = cv2.imread(file)  # 读取数据
            List.append(img_cv)
            id=getId(file)
            lableList.append(id)
        else:
            readFile(file)
def nom(x):
    x_mean = np.mean(x)
    x_var = np.var(x)
    x_normalized = (x - x_mean) / np.sqrt(x_var)
    return x_normalized
def getData(List1):
    trainX,trainY,testX,testY=[],[],[],[]
    arr=[i for i in range(0,len(List1))]
    randomArr=np.random.choice(arr, int(len(List1)*0.2), replace=False)
    last=[]
    for i  in range(0,len(List1)) :
        if(i not in randomArr):
            last.append(i)
    List1=np.array(List1)
    List1=nom(List1)
    lableList1=np.array(lableList)
    trainX=List1[last]
    trainY=lableList1[last]
    testX=List1[randomArr]
    testY=lableList1[randomArr]


    train_dataset = Data.TensorDataset(torch.tensor(trainX, dtype=torch.float32), torch.tensor(trainY, dtype=torch.float32))
    test_dataset = Data.TensorDataset(torch.tensor(testX, dtype=torch.float32), torch.tensor(testY, dtype=torch.float32))


    return train_dataset,test_dataset
def train(trainData,testData):
    train_loader = Data.DataLoader(dataset=trainData,
                                  batch_size=128,
                                  shuffle=True,  # 不打乱顺序,便于查看
                                  num_workers=2)
    model=CNNnet()
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_count = []
    accArr=[]
    for epoch in range(10):
        for i,(batch_x,batch_y) in enumerate(train_loader):
            out = model(batch_x)
            loss = loss_func(out,batch_y.long())
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            loss_count.append(loss.item())
            acc=test(model,testData)
            accArr.append(acc)

    torch.save(model, r'./model/cnn')
    plt.figure('acc')
    plt.plot(accArr,label='acc')
    plt.legend()
    plt.show()
def test(model,testData):
    # 测试网络
    #model = torch.load(r'./model/cnn')
    accuracy_sum = []
    test_loader = Data.DataLoader(dataset=testData,
                                  batch_size=32,
                                  shuffle=False,  # 不打乱顺序,便于查看
                                  num_workers=2)
    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(test_loader):
            out = model(test_x)
            accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
            accuracy_sum.append(accuracy.mean())

    return sum(accuracy_sum) / len(accuracy_sum)


if __name__ == '__main__':
    path=r'./faces_4'
    readFile(path)
    trainData,testData=getData(List)
    train(trainData,testData)
    #test(testData)