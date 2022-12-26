from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import os
import pandas as pd
def readFile(path):
    global total_photo
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        file = os.path.join(path, file)
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            photo = imgplt.imread(file)
            photo = photo.reshape(1, -1)
            photo = pd.DataFrame(photo)
            total_photo = total_photo.append(photo, ignore_index=True)
        else:
            readFile(file)
def kmeans():
    clf = KMeans(n_clusters=4)
    clf.fit(total_photo)
    y_predict = clf.predict(total_photo)
    centers = clf.cluster_centers_
    result = centers[y_predict]
    result = result.astype("int64")
    result = result.reshape(-1, 30, 32)
    return result,y_predict

def draw():
    fig,ax  = plt.subplots(nrows=4,ncols=20,sharex = True,sharey = True,figsize = [15,8])
    plt.subplots_adjust(wspace = 0,hspace = 0)
    num=[0for i in range(10)]
    for i in range(624):
        if(num[y_predict[i]]>=20) :continue
        ax[y_predict[i],num[y_predict[i]]].imshow(result[i],cmap = 'gray')
        num[y_predict[i]]+=1
    plt.xticks([])
    plt.yticks([])
    plt.show()
if __name__ == '__main__':
    total_photo = pd.DataFrame()
    readFile("./faces_4")
    total_photo = total_photo.values
    result,y_predict = kmeans()
    draw()

