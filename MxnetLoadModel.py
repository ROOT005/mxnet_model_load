#--coding:utf-8--
import mxnet as mx

#下载模型
# print("下载数据")
# path='http://data.mxnet.io/models/imagenet/'
# [mx.test_utils.download(path+'resnet/200-layers/resnet-200-0000.params'),
#  mx.test_utils.download(path+'resnet/200-layers/resnet-200-symbol.json'),
#  mx.test_utils.download(path+'synset.txt')]

# print("加载到cpu")
ctx = mx.cpu()

sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-200', 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

import matplotlib.pyplot as plt
import numpy as np
#定义一个简单的data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(url, show=False):
    # 下载或者打开图片
    fname = mx.test_utils.download(url, fname=url.split('/')[-1].split('?')[0])
    img = mx.image.imread(fname)
    if img is None:
        return None
    if show:
        plt.imshow(img.asnumpy())
        plt.axis('off')
    #格式化为(batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    img = img.expand_dims(axis=0) # batchify
    return img

def predict(url):
    print("开始预测")
    img = get_image(url, show=True)
    #计算并预测概率
    mod.forward(Batch([img]))
    prob = mod.get_outputs()[0].asnumpy()
    #从前五个中找出最大值
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    probability = 0
    label = ''
    for i in a[0:5]:
        if prob[i] > probability:
            probability = prob[i]
            label = labels[i]
    name = ' '
    name = name.join(label.split(',')[1:])
    plt.text(125, -12,  '名称：'+name+",\n概率："+str(probability), fontsize=8,fontdict={'family':'NoTo Sans SC', 'color':'red'});
    plt.show()
    print('概率=%f, 种类=%s' %(probability, name))

predict('dog.jpg')
predict('cat.jpg')
predict('keybord.jpg')
predict('pig.png')
#predict('gpang.jpg')

predict('minion.jpg')
