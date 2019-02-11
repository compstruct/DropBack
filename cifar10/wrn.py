"""
Modified from https://github.com/mitmul/chainer-cifar10
"""
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
import numpy as np

class WideBasic(chainer.Chain):
    def __init__(self, n_input, n_output, stride, dropout, num=0):
        self.dtype = np.float32
        W = initializers.HeNormal()
        super(WideBasic, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                n_input, n_output, 3, stride, 1, nobias=True, initialW=W)
            self.conv2 = L.Convolution2D(
                n_output, n_output, 3, 1, 1, nobias=True, initialW=W)
            self.bn1 = L.BatchNormalization(n_input)
            self.bn2 = L.BatchNormalization(n_output)
            if n_input != n_output:
                self.shortcut = L.Convolution2D(
                    n_input, n_output, 1, stride, nobias=True, initialW=W)
        self.dropout = dropout
        self.num = num
        self.acts = {}

    def __call__(self, x):
        x = F.relu(self.bn1(x))
        self.acts['bn1_wb'] = x
        h = F.relu(self.bn2(self.conv1(x)))
        self.acts['conv1_wb'] = h
        if self.dropout:
            h = F.dropout(h)
        h = self.conv2(h)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        return h + shortcut


class WideBlock(chainer.ChainList):
    def __init__(self, n_input, n_output, count, stride, dropout):
        super(WideBlock, self).__init__()
        num = 0
        self.add_link(WideBasic(n_input, n_output, stride, dropout, num=num))
        for _ in range(count - 1):
            num+=1
            self.add_link(WideBasic(n_output, n_output, 1, dropout, num=num))

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class WideResNet(chainer.Chain):

    insize = 224

    def __init__(
            self, widen_factor=10, depth=28, n_classes=10, dropout=True):
        k = widen_factor
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        n_stages = [16, 16 * k, 32 * k, 64 * k]
        self.dtype = np.float32
        W = initializers.HeNormal()
        bias = initializers.Zero(self.dtype)
        super(WideResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, n_stages[0], 3, 1, 1, nobias=True, initialW=W)
            self.wide2 = WideBlock(n_stages[0], n_stages[1], n, 1, dropout)
            self.wide3 = WideBlock(n_stages[1], n_stages[2], n, 2, dropout)
            self.wide4 = WideBlock(n_stages[2], n_stages[3], n, 2, dropout)
            self.bn5 = L.BatchNormalization(n_stages[3])
            self.fc6 = L.Linear(n_stages[3], n_classes, initialW=W)

    def __call__(self, x, t=None):
        x = F.cast(x, self.dtype)
        h = self.conv1(x)
        h = self.wide2(h)
        h = self.wide3(h)
        h = self.wide4(h)
        h = F.relu(self.bn5(h))
        h = F.average_pooling_2d(h, (h.shape[2], h.shape[3]))
        h =  self.fc6(h)
        if t is None:
            return h
        else:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    model = WideResNet(10)
    y = model(x)
