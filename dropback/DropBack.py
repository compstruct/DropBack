"""
An emulation of the DropBack algorithm using Chainer.

https://arxiv.org/abs/1806.06949

Author: Maximilian Golub
"""

import chainer
from chainer import cuda

import time
import os


class DropBack(chainer.training.StandardUpdater):

    def __init__(self, train, optimizer, output_dir, converter=chainer.dataset.convert.concat_examples, device=-1,
                 tracked_size=0, freeze=False,
                 decay_init=False, **kwargs):
        """
        Create a new DropBack Updater. The most important parameter is the tracked size, which controls
        how many parameters are retained.

        :param train: The train iterator
        :param optimizer: The optimizer
        :param output_dir: Directory to output data to (init params, stats...)
        :param converter: Chainer dataset converter
        :param tracked_size: The number of params to track
        :param freeze: Epoch to freeze the tracked selection on
        :param decay_init: True/False: Decay the initial parameters every iteration
        :param kwargs:
        """
        super(DropBack, self).__init__(train, optimizer, converter=converter, device=device, **kwargs)
        self.opt = self.get_optimizer('main')
        self.tracked_size = tracked_size
        self.first_iter = True
        self.init_params = []
        self.output_dir = output_dir
        try:
            os.makedirs(self.output_dir)
        except OSError:
            pass
        self.time_stamp = time.time()
        self.params = None
        self.train = train
        self.freeze = False
        self.use_freeze = freeze
        self.frozen_masks = [None]
        self.decay_init = decay_init
        self.track = 100
        self.xp = cuda.get_array_module(next(self.opt.target.params()))

    def update(self):
        """
        Where the magic happens. Finds a threshold that will limit the number of params in the network
        to the tracked_size, and resets those params to the initial value to emulate how DropBack would
        work in real hardware.

        Chainer will calculate all grads, and this updater inserts itself before the next
        forward pass can occur to set the parameters back to what they should be. Only the params with the largest
        current-initial value will not be reset to initial. This emulates the accumulated gradient updates of the actual
        algorithm.
        :return:
        """
        xp = self.xp
        super(DropBack, self).update()
        if self.first_iter:
            self.first_iter = False
            self.params = [i for i in self.opt.target.params()]
            for i, p in enumerate(self.params):
                self.init_params.append(xp.copy(p.data))
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            xp.savez(os.path.join(self.output_dir, 'init_params_{0}'.format(self.time_stamp)),
                    self.init_params)
            if self.tracked_size:
                self.frozen_masks = [None] * len(self.params)


        if self.decay_init and not self.first_iter:
            for i, _ in enumerate(self.init_params):
                self.init_params[i] = self.init_params[i]*.90
        if self.tracked_size:
            if not self.freeze:
                abs_values = []
                for i, param in enumerate(self.params):
                    if param.name == 'b':
                        values = (xp.abs(param.data).flatten()).copy()
                    else:
                        values = (xp.abs(param.data - self.init_params[i]).flatten()).copy()
                    abs_values.append(values)
                abs_vals = xp.concatenate(abs_values)
                thresh = xp.sort(abs_vals)[-self.tracked_size]
            for i, param in enumerate(self.params):
                if param.name == 'b':
                    if self.freeze:
                        mask = self.frozen_masks[i]
                    else:
                        mask = xp.abs(param.data) > thresh
                    param.data = mask*param.data
                else:
                    if self.freeze:
                        mask = self.frozen_masks[i]
                    else:
                        mask = xp.abs(param.data - self.init_params[i]) > thresh
                    param.data = mask*param.data + self.init_params[i]*~mask
                self.frozen_masks[i] = mask
            if self.iteration == 3465 and self.device >= 0:
                print("Checking inv...")
                total_sum = sum([xp.count_nonzero(p.data != self.init_params[i]) for i, p in enumerate(self.params)])
                print("********\n\n Total non zero is: {}\n\n1*********".format(total_sum))
                assert total_sum <= self.tracked_size * 1.1
        if self.track:
            if self.iteration-1 % 100 == 0:
                flat_now = xp.concatenate([i.array.ravel() for i in self.params])
                flat_0 = xp.concatenate([i.ravel() for i in self.init_params])
                xp.savez(os.path.join(self.output_dir, f'l2_{self.iteration-1}'), xp.linalg.norm(flat_now - flat_0))
                xp.savez(os.path.join(self.output_dir, f'param_hist_{self.iteration-1}'), xp.concatenate([i.array.ravel() for i in self.params if i.name == 'b' or i.name == 'W']))

    def serialize(self, serializer):
        super(DropBack, self).serialize(serializer)
        self.opt = serializer('opt', self.get_optimizer('main'))
        self.tracked_size = serializer('max_cache', self.tracked_size)
        self.first_iter = serializer('first_iter', self.first_iter)
        self.init_params = serializer('init_params', self.init_params)
        self.output_dir = serializer('output_dir', self.output_dir)
        self.time_stamp = serializer('time_stamp', self.time_stamp)
        self.params = serializer('params', self.params)
        self.train = serializer('train', self.train)
        self.freeze = serializer('freeze', self.freeze)
        self.use_freeze = serializer('use_freeze', self.use_freeze)
        self.frozen_masks = serializer('frozen_masks', self.frozen_masks)
        self.decay_init = serializer('decay_init', self.decay_init)
        self.fired_assert = serializer('fired_assert', self.fired_assert)