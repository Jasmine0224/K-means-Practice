import numpy as np
import PIL.Image as image
from sklearn import preprocessing
from sklearn.cluster import KMeans
from skimage import color

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
kmeans = KMeans(n_clusters=16)
label = kmeans.fit_predict(img)
label = label.reshape([width, height])

# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label)*255).astype(np.uint8)
label_color = label_color.transpose(1, 0, 2)
images = image.fromarray(label_color)
images.save('weixin_mark_color.jpg')
