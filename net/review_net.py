import chainer
import chainer.functions as F
import chainer.links as L


class ReviewNet(chainer.Chain):
    def __init__(self, n_class, in_ch):
        super().__init__(
        
        )
        self.train = True
        self.n_class = n_class
        self.active_learn = False

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        x.volatile = not self.train

        self.prob = F.softmax(self.h)
        self.loss = F.softmax_cross_entropy(self.h, t)
        self.accuracy = F.accuracy(self.h, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)

        return self.loss
