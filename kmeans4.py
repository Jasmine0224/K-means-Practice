import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import PIL.Image as image

# 加载图像，并规范化图像数据
def load_data(filepath):
    f = open('./baby.jpg', 'rb')
    img = image.open(f)
    data = []

    width, height = img.size
    for x in range(width):
        for y in range(height):
            c1, c2, c3 = img.getpixel((x,y))
            data.append([(c1+1)/256, (c2+1)/256, (c3+1)/256])
    
    f.close()
    return np.mat(data), width, height

# 加载图像，得到规范化的img数据，以及图片尺寸
img, width, height = load_data('kmeans-master/baby.jpg')

# 使用k-means对图像进行16聚类
kmeans = KMeans(n_clusters=16)
label = kmeans.fit_predict(img)
label = label.reshape([width, height])

# 生成一个新图像用来保存图像聚类压缩后的结果
img = image.new('RGB', (width, height))
for x in range(width):
    for y in range(height):
        c1 = kmeans.cluster_centers_[label[x, y], 0]
        c2 = kmeans.cluster_centers_[label[x, y], 1]
        c3 = kmeans.cluster_centers_[label[x, y], 2]
        img.putpixel((x, y), (int(c1*256)-1, int(c2*256)-1, int(c3*256)-1))
img.save('baby_new.jpg')
