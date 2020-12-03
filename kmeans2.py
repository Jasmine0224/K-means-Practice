import numpy as np
import PIL.Image as image
from sklearn import preprocessing
from sklearn.cluster import KMeans

def load_data(filepath):
    # 加载图像，并对数据进行规范化

    f = open(filepath, 'rb')
    img = image.open(f)
    width, height = img.size

    data = []
    for x in range(width):
        for y in range(height):
            c1, c2, c3 = img.getpixel((x,y))
            data.append([c1, c2, c3])
    
    f.close()

    # 采用Min_Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)

    return np.mat(data), width, height

# 加载图像并得到规范化的结果img, 和图像尺寸
img, width, height = load_data('kmeans-master/weixin.jpg')

# 用k_means对图像进行16聚类
kmeans = KMeans(n_clusters=2)
label = kmeans.fit_predict(img)
label = label.reshape([width, height])

# 创建新图像用来表示聚类的结果，并设置灰度
pic_mark = image.new('L', (width, height))
for x in range(width):
    for y in range(height):
        pic_mark.putpixel((x,y), int(256/(label[x][y]+1)-1))
pic_mark.save('weixin_mark.jpg', 'JPEG')