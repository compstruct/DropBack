"""
Modified from https://github.com/soskek/variational_dropout_sparsifies_dnn
"""
import chainer
from chainer import cuda
from chainer import function
from chainer import functions as F
from chainer import utils
from chainer import configuration
import warnings
from collections import defaultdict
import chainer.links as L
from chainer import reporter
import numpy
from scipy import sparse

try:
    import cupy
except ImportError:
    import numpy as cupy

from cifar10 import vgg

class SparseLinearForwardCPU(chainer.links.Linear):

    def __init__(self, old_linear, W_mask=None, with_dense=False):
        W = old_linear.W.data
        b = getattr(old_linear, 'b', None)
        super(SparseLinearForwardCPU, self).__init__(
            W.shape[1], W.shape[0])
        self.W.data[:] = self.xp.array(W)
        if b is not None:
            b = b.data
            self.b.data[:] = self.xp.array(b)
        if not with_dense:
            delattr(self, 'W')
            if b is not None:
                delattr(self, 'b')

        xp = cuda.get_array_module(W)
        if W_mask is None:
            W_mask = xp.ones(W.shape).astype('f')

        if xp is numpy:
            self.sparse_W = sparse.csc_matrix(W * W_mask)
            if b is not None:
                self.sparse_b = numpy.array(b).astype('f')
        else:
            self.sparse_W = sparse.csr_matrix(
                xp.asnumpy(W) * xp.asnumpy(W_mask))
            if b is not None:
                self.sparse_b = xp.asnumpy(b)[None, ]

    def __call__(self, x):
        train = configuration.config.train
        if self.xp is numpy and not train:
            if isinstance(x, chainer.Variable):
                x = x.data
            if x.ndim > 2:
                x = x.reshape(x.shape[0], x.size // x.shape[0])
            return self.sparse_W.dot(x.T).T.astype('f') + \
                getattr(self, 'sparse_b', 0.)
        else:
            warnings.warn('SparseLinearForwardCPU link is made for'
                          ' inference usage. Sparse computation'
                          ' (scipy.sparse) computation is used'
                          ' only in inference mode'
                          ' rather than training mode.')
            if hasattr(self, 'W'):
                return super(SparseLinearForwardCPU, self).__call__(x)
            else:
                NotImplementedError

def compositional_calculate_kl(W, log_sigma2, loga_threshold=3.,
                               eps=1e-8, thresholds=(-8., 8.)):

    def _calculate_kl(W, log_sigma2):
        log_alpha = F.clip(log_sigma2 - F.log(W ** 2 + 1e-8), -8., 8.)
        clip_mask = (log_alpha.data > loga_threshold)
        normalizer = 1. / W.size
        reg = (0.63576 * F.sigmoid(1.87320 + 1.48695 * log_alpha)) + \
              (- 0.5 * F.log1p(F.exp(- log_alpha))) - 0.63576
        xp = cuda.get_array_module(reg)
        reg = F.where(clip_mask, xp.zeros(reg.shape).astype('f'), reg)
        return - F.sum(reg) * normalizer

    return _calculate_kl(W, log_sigma2)


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


def _grad_sigmoid(x):
    return x * (1 - x)


class KL(function.Function):

    def __init__(self, clip_mask):
        self.clip_mask = clip_mask

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        log_alpha = inputs[0]
        reg = (0.63576 * _sigmoid(1.87320 + 1.48695 * log_alpha)) + \
              (- 0.5 * numpy.log1p(numpy.exp(- log_alpha))) - 0.63576
        reg = reg * (1. - self.clip_mask)
        reg = - reg.sum() / log_alpha.size
        reg = utils.force_array(reg, log_alpha.dtype)
        return reg,

    def backward_cpu(self, inputs, gy):
        log_alpha = inputs[0]
        gy = gy[0]

        sig = _sigmoid(1.87320 + 1.48695 * log_alpha)
        exp_m_log_alpha = numpy.exp(- log_alpha)

        reg = (0.63576 * sig) + \
              (- 0.5 * numpy.log1p(exp_m_log_alpha)) - 0.63576
        reg = reg * (1. - self.clip_mask)

        greg = - gy / log_alpha.size * reg

        gla_from_1 = greg * 0.63576 * _grad_sigmoid(sig) * 1.48695
        gla_from_2 = greg * \
            (- 0.5) / (1. + exp_m_log_alpha) * exp_m_log_alpha

        gla = gla_from_1 + gla_from_2
        gla = utils.force_array(gla, log_alpha.dtype)
        return gla,

    def forward_gpu(self, inputs):
        log_alpha = inputs[0]
        reg = cuda.elementwise(
            'T la, T clip',
            'T reg',
            '''
            const T half = 0.5;
            const T c063576 = 0.63576;
            reg = (c063576 * 
                   (tanh(((T)1.87320 + (T)1.48695 * la) * half) * half + half) 
                   - half * log1p(exp(-la)) - c063576) * ((T)1.0 - clip);
            ''',
            'kl_fwd')(
                log_alpha, self.clip_mask)
        reg = utils.force_array(- reg.sum() / log_alpha.size, log_alpha.dtype)
        return reg,

    def backward_gpu(self, inputs, gy):
        log_alpha = inputs[0]
        gy = gy[0]

        gla = cuda.elementwise(
            'T gy, T la, T clip',
            'T gla',
            '''
            const T half = 0.5;
            const T c1 = 1.0;
            const T c063576 = 0.63576;
            const T c148695 = 1.48695;
            T sig = (tanh((1.87320 + c148695 * la) * half) * half + half);
            T exp_m_la = exp(- la);
            T reg = (c063576 * sig - 
                     half * log1p(exp_m_la) - c063576) * (c1 - clip);
            T greg = - gy * reg;
            gla = greg * (c063576 * (sig * (c1 - sig)) * c148695
                          - half / (c1 + exp_m_la) * exp_m_la)
            ''',
            'kl_bwd')(
                (gy / log_alpha.size).astype(log_alpha.dtype),
                log_alpha, self.clip_mask)
        return gla,


def calculate_kl(W=None, loga_threshold=3.,
                 log_sigma2=None, log_alpha=None,
                 eps=1e-8, thresholds=(-8., 8.)):
    if log_alpha is None:
        if log_sigma2 is None or W is None:
            AttributeError()
        log_alpha = calculate_log_alpha(
            W, log_sigma2, eps=eps, thresholds=thresholds)
    clip_mask = (log_alpha.data > loga_threshold).astype(
        log_alpha.data.dtype, copy=False)
    return KL(clip_mask)(log_alpha)


class LogAlpha(function.Function):
    """Function calculate log alpha from W and log sigma^2.

    This function is memory-efficient by recomputing in backward.
    """

    def __init__(self, eps=1e-8, lower_threshold=-8., upper_threshold=8.):
        self.eps = eps
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        W, log_sigma2 = inputs
        log_alpha = log_sigma2 - numpy.log(numpy.square(W) + self.eps)
        log_alpha = utils.force_array(
            numpy.minimum(numpy.maximum(
                self.lower_threshold, log_alpha), self.upper_threshold),
            W.dtype)
        return log_alpha,

    def backward_cpu(self, inputs, gy):
        W, log_sigma2 = inputs
        gy = gy[0]
        square_W = numpy.square(W) + self.eps
        log_alpha = log_sigma2 - numpy.log(square_W)
        clip = (self.lower_threshold < log_alpha) * \
               (log_alpha < self.upper_threshold)
        clip_gy = gy * clip
        gs = clip_gy
        gW = - clip_gy / square_W * 2. * W
        gs = utils.force_array(gs, log_sigma2.dtype)
        gW = utils.force_array(gW, W.dtype)
        return gW, gs

    def forward_gpu(self, inputs):
        W, log_sigma2 = inputs
        return cuda.elementwise(
            'T W, T ls, T eps, T lo_th, T up_th',
            'T y',
            'y = min(max(ls - log(W * W + eps), lo_th), up_th)',
            'log_alpha_fwd')(
                W, log_sigma2,
                self.eps, self.lower_threshold, self.upper_threshold),

    def backward_gpu(self, inputs, gy):
        W, log_sigma2 = inputs
        gy = gy[0]
        gW, gs = cuda.elementwise(
            'T W, T ls, T gy, T eps, T lo_th, T up_th',
            'T gW, T gs',
            '''
            T square_W = W * W + eps;
            T y = ls - log(square_W);
            gs = ((y > lo_th) & (y < up_th))? gy : (T)0;
            gW = - gs / square_W * 2 * W;
            ''',
            'log_alpha_bwd')(
                W, log_sigma2, gy,
                self.eps, self.lower_threshold, self.upper_threshold)
        return gW, gs


def calculate_log_alpha(W, log_sigma2, eps=1e-8, thresholds=(-8., 8.)):
    lower_threshold, upper_threshold = thresholds
    return LogAlpha(eps, lower_threshold, upper_threshold)(W, log_sigma2)


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

# TODO: RNNVDLinear: efficient multiple (sampled) Linear re-using calculations.


class VDLinear(function.Function):
    """Linear function using variational dropout.

    This function is memory-efficient by recomputing in backward.
    """

    def __init__(self, clip_mask,
                 eps=1e-8, lower_threshold=-8., upper_threshold=8.):
        self.clip_mask = clip_mask
        self.eps = eps
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        # TODO: merge calculate_log_alpha and reuse W ** 2
        x, W, log_alpha = inputs[:3]
        x = _as_mat(x)
        W = ((1. - self.clip_mask) * W).astype(W.dtype, copy=False)
        mu = x.dot(W.T)
        si = numpy.sqrt(
            (x * x).dot((numpy.exp(log_alpha) * W * W).T) + self.eps)
        self.normal_noise = numpy.random.standard_normal(mu.shape).astype(
            x.dtype, copy=False)
        y = mu + si * self.normal_noise
        if len(inputs) == 4:
            b = inputs[3]
            y += b
        return y,

    def forward_gpu(self, inputs):
        # TODO: cuda kernel
        # TODO: merge calculate_log_alpha and reuse W ** 2
        x, W, log_alpha = inputs[:3]
        x = _as_mat(x)
        W, alpha_W2 = cuda.elementwise(
            'T W, T clip_mask, T la',
            'T clip_W, T alpha_W2',
            '''
            clip_W = ((T)1.0 - clip_mask) * W;
            alpha_W2 = exp(la) * clip_W * clip_W;
            ''',
            'vdl1_fwd')(
                W, self.clip_mask, log_alpha)
        x2 = x * x
        mu = x.dot(W.T)
        si2 = x2.dot(alpha_W2.T)
        self.normal_noise = cupy.random.standard_normal(mu.shape).astype(
            x.dtype, copy=False)
        y = cuda.elementwise(
            'T mu, T si2, T eps, T noise',
            'T y',
            '''
            y = mu + sqrt(si2 + eps) * noise;
            ''',
            'vdl2_fwd')(
                mu, si2, self.eps, self.normal_noise)
        if len(inputs) == 4:
            b = inputs[3]
            y += b
        return y,

    def backward_cpu(self, inputs, gy):
        # TODO: merge calculate_log_alpha and reuse W ** 2
        x, W, log_alpha = inputs[:3]
        x = _as_mat(x)
        gy = gy[0]

        clip = (1. - self.clip_mask).astype(W.dtype, copy=False)
        clip_W = clip * W
        x2 = x * x
        W2 = clip_W * clip_W
        alpha = numpy.exp(log_alpha)
        alpha_W2 = W2 * alpha
        si_before_sqrt = x2.dot(alpha_W2.T) + self.eps

        gmu = gy
        gx_from_gmu = gmu.dot(clip_W)
        gW_from_gmu = gmu.T.dot(x) * clip

        gsi = gy * self.normal_noise
        gsi_before_sqrt = gsi * (0.5 / numpy.sqrt(si_before_sqrt))
        gx2_from_gsi = gsi_before_sqrt.dot(alpha_W2)
        gx_from_gsi = gx2_from_gsi * (2. * x)
        galpha_W2_from_gsi = gsi_before_sqrt.T.dot(x2)
        gW2_from_gsi = galpha_W2_from_gsi * alpha
        gW_from_gsi = gW2_from_gsi * (2. * clip_W)
        galpha_from_gsi = galpha_W2_from_gsi * W2
        glog_alpha = galpha_from_gsi * numpy.exp(log_alpha)

        gx = (gx_from_gmu + gx_from_gsi).astype(x.dtype, copy=False).reshape(
            inputs[0].shape)
        gW = (gW_from_gmu + gW_from_gsi).astype(W.dtype, copy=False)
        if len(inputs) == 4:
            gb = gy.sum(0)
            return gx, gW, glog_alpha, gb
        else:
            return gx, gW, glog_alpha

    def backward_gpu(self, inputs, gy):
        # TODO: merge calculate_log_alpha and reuse W ** 2
        x, W, log_alpha = inputs[:3]
        x = _as_mat(x)
        xp = cuda.get_array_module(x)
        gy = gy[0]

        clip, clip_W, W2, alpha, alpha_W2 = cuda.elementwise(
            'T W, T clip_mask, T la',
            'T clip, T clip_W, T W2, T alpha, T alpha_W2',
            '''
            clip = ((T)1.0 - clip_mask);
            clip_W = clip * W;
            W2 = clip_W * clip_W;
            alpha = exp(la);
            alpha_W2 = W2 * alpha;
            ''',
            'vdl1_bwd')(
                W, self.clip_mask, log_alpha)

        x2 = x * x
        si_before_sqrt = x2.dot(alpha_W2.T)
        gx_from_gmu = gy.dot(clip_W)
        gW_from_gmu = gy.T.dot(x) * clip

        gsi_before_sqrt = cuda.elementwise(
            'T gy, T noise, T si_bf_sqrt, T eps',
            'T gsi_bf_sqrt',
            '''
            gsi_bf_sqrt = gy * noise * ((T)0.5 / sqrt(si_bf_sqrt + eps));
            ''',
            'gsi_bwd')(
                gy, self.normal_noise, si_before_sqrt, self.eps)

        galpha_W2_from_gsi = gsi_before_sqrt.T.dot(x2)
        gW_from_gsi = galpha_W2_from_gsi * alpha * (2. * clip_W)
        glog_alpha = galpha_W2_from_gsi * W2 * alpha

        gW, glog_alpha = cuda.elementwise(
            'T galpha_W2_from_gsi, T alpha, T clip_W, T W2, T gW_from_gmu',
            'T gW, T glog_alpha',
            '''
            gW = galpha_W2_from_gsi * alpha * (T)2.0 * clip_W + gW_from_gmu;
            glog_alpha = galpha_W2_from_gsi * W2 * alpha;
            ''',
            'gW_glog_bwd')(
                galpha_W2_from_gsi, alpha, clip_W, W2, gW_from_gmu)

        gx_from_gsi = gsi_before_sqrt.dot(alpha_W2) * 2. * x
        gx = (gx_from_gmu + gx_from_gsi).astype(x.dtype, copy=False).reshape(
            inputs[0].shape)
        if len(inputs) == 4:
            gb = gy.sum(0)
            return gx, gW, glog_alpha, gb
        else:
            return gx, gW, glog_alpha


def vd_linear(x, W, b, loga_threshold=3., log_sigma2=None,
              log_alpha=None, eps=1e-8, thresholds=(-8., 8.)):
    if log_alpha is None:
        if log_sigma2 is None:
            AttributeError()
        log_alpha = calculate_log_alpha(
            W, log_sigma2, eps=eps, thresholds=thresholds)
    clip_mask = (log_alpha.data > loga_threshold).astype(
        log_alpha.data.dtype, copy=False)

    train = configuration.config.train
    if train:
        if b is None:
            return VDLinear(clip_mask, eps)(
                x, W, log_alpha)
        else:
            return VDLinear(clip_mask, eps)(
                x, W, log_alpha, b)
    else:
        return F.linear(x, (1. - clip_mask) * W, b)

# try:
#     import variational_dropout_sparsifies_dnn.sparse_chainer as sparse_chainer
#     import variational_dropout_sparsifies_dnn.vd_functions as VDF
# except ImportError:
#     import sparse_chainer as sparse_chainer
#     import vd_functions as VDF

# Memo: If p=0.95, then alpha=19. ln(19) = 2.94443897917.
#       Thus, log_alpha_threashold is set 3.0 approximately.

configuration.config.user_memory_efficiency = 0
# 1<= : simple calculation like an element-wise one
# 2<= : little simple calculation like a series of element-wise ones
# 3<= : complex calculations like matrix ones
# more memory efficient, it takes much time

P_THRESHOLD = 0.95
LOGA_THRESHOLD = 3.
INITIAL_LOG_SIGMA2 = chainer.initializers.Constant(-10.)


def get_vd_links(link):
    if isinstance(link, chainer.Chain):
        for child_link in link.links(skipself=True):
            for vd_child_link in get_vd_links(child_link):
                yield vd_child_link
    else:
        if getattr(link, 'is_variational_dropout', False):
            yield link


def calculate_p(link):
    """Calculate (something like) probabilities of variational dropout
    This method takes high computational cost.
    """
    alpha = link.xp.exp(calculate_log_alpha(
        link.W, link.log_sigma2, eps=1e-8, thresholds=(-8., 8.)).data)
    p = alpha / (1 + alpha)
    return p


def calculate_stats(chain, threshold=P_THRESHOLD):
    """Calculate stats for parameters of variational dropout
    This method takes high computational cost.
    """
    xp = chain.xp
    stats = {}
    all_p = [calculate_p(link).flatten()
             for link in get_vd_links(chain)]
    if not all_p:
        return defaultdict(float)
    all_p = xp.concatenate(all_p, axis=0)
    stats['mean_p'] = xp.mean(all_p)

    all_threshold = [link.p_threshold
                     for link in get_vd_links(chain)]
    if any(th != threshold for th in all_threshold):
        warnings.warn('The threshold for sparsity calculation'
                      ' is different from'
                      ' thresholds used for prediction with'
                      ' threshold-based pruning.')
    # TODO: directly use threshold of each link

    is_zero = (all_p > threshold)
    stats['sparsity'] = xp.mean(is_zero)

    n_non_zero = (1 - is_zero).sum()
    if n_non_zero == 0:
        stats['W/Wnz'] = float('inf')
    else:
        stats['W/Wnz'] = all_p.size * 1. / n_non_zero
    return stats


class VariationalDropoutLinear(chainer.links.Linear):

    def __init__(self, in_size, out_size, nobias=False,
                 initialW=None, initial_bias=None,
                 p_threshold=P_THRESHOLD, loga_threshold=LOGA_THRESHOLD,
                 initial_log_sigma2=INITIAL_LOG_SIGMA2):
        super(VariationalDropoutLinear, self).__init__(
            in_size, out_size, nobias=nobias,
            initialW=initialW, initial_bias=initial_bias)
        self.add_param('log_sigma2', initializer=initial_log_sigma2)
        if in_size is not None:
            self._initialize_params(in_size, log_sigma2=True)
        self.p_threshold = p_threshold
        self.loga_threshold = loga_threshold
        self.is_variational_dropout = True
        self.is_variational_dropout_linear = True

    def _initialize_params(self, in_size, log_sigma2=False):
        if not log_sigma2:
            self.W.initialize((self.out_size, in_size))
        else:
            self.log_sigma2.initialize((self.out_size, in_size))

    def get_sparse_cpu_model(self):
        log_alpha = calculate_log_alpha(
            self.W, self.log_sigma2, eps=1e-8, thresholds=(-8., 8.))
        clip_mask = (log_alpha.data > self.loga_threshold)
        return SparseLinearForwardCPU(self, (1. - clip_mask))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        if self.log_sigma2.data is None:
            self._initialize_params(x.size // x.shape[0], log_sigma2=True)
        return vd_linear(
            x, self.W, self.b, self.loga_threshold, log_sigma2=self.log_sigma2,
            log_alpha=None, eps=1e-8, thresholds=(-8., 8.))


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class VariationalDropoutConvolution2D(chainer.links.Convolution2D):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 p_threshold=P_THRESHOLD, loga_threshold=LOGA_THRESHOLD,
                 initial_log_sigma2=INITIAL_LOG_SIGMA2):
        super(VariationalDropoutConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad,
            nobias=nobias, initialW=initialW,
            initial_bias=initial_bias)

        self.add_param('log_sigma2', initializer=initial_log_sigma2)
        if in_channels is not None:
            self._initialize_params(in_channels, log_sigma2=True)
        self.p_threshold = p_threshold
        self.loga_threshold = loga_threshold
        self.is_variational_dropout = True

    def _initialize_params(self, in_channels, log_sigma2=False):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        if not log_sigma2:
            self.W.initialize(W_shape)
        else:
            self.log_sigma2.initialize(W_shape)

    def dropout_convolution_2d(self, x):
        train = configuration.config.train
        W, b = self.W, self.b
        log_alpha = calculate_log_alpha(
            self.W, self.log_sigma2, eps=1e-8, thresholds=(-8., 8.))
        clip_mask = (log_alpha.data > self.loga_threshold)
        if train:
            W = (1. - clip_mask) * W
            mu = F.convolution_2d(x, (1. - clip_mask) * W, b=None,
                                  stride=self.stride, pad=self.pad)
            si = F.sqrt(
                F.convolution_2d(x * x, F.exp(log_alpha) * W * W, b=None,
                                 stride=self.stride, pad=self.pad) + 1e-8)
            normal_noise = self.xp.random.normal(
                0., 1., mu.shape).astype('f')
            activation = mu + si * normal_noise
            return F.bias(activation, b)
        else:
            return F.convolution_2d(x, (1. - clip_mask) * W, b,
                                    stride=self.stride, pad=self.pad)

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return self.dropout_convolution_2d(x)


class VariationalDropoutTanhRNN(chainer.Chain):

    def __init__(self, in_size, out_size, nobias=False,
                 initialW=None, initial_bias=None,
                 p_threshold=P_THRESHOLD, loga_threshold=LOGA_THRESHOLD,
                 initial_log_sigma2=INITIAL_LOG_SIGMA2):
        W = VariationalDropoutLinear(
            in_size + out_size, out_size, nobias=nobias,
            initialW=initialW, initial_bias=initial_bias,
            p_threshold=p_threshold, loga_threshold=loga_threshold,
            initial_log_sigma2=initial_log_sigma2)
        super(VariationalDropoutTanhRNN, self).__init__(W=W)
        self.in_size = in_size
        self.out_size = out_size
        self.h = None

    def reset_state(self):
        self.h = None

    def set_state(self, h):
        self.h = h

    def __call__(self, x, h=None):
        """RNN call
        If h is given, this works as stateless rnn.
        Otherwise, stateful rnn.
        """
        stateful = (h is None)
        if stateful:
            if self.h is None:
                self.h = self.xp.zeros((x.shape[0], self.out_size)).astype('f')
            h = self.h
        new_h = F.tanh(self.W(F.concat([x, h], axis=1)))

        if stateful:
            self.h = new_h
        else:
            self.h = None

        return new_h


class VariationalDropoutLSTM(chainer.Chain):

    def __init__(self, in_size, out_size, nobias=False,
                 initialW=None, initial_bias=None,
                 p_threshold=P_THRESHOLD, loga_threshold=LOGA_THRESHOLD,
                 initial_log_sigma2=INITIAL_LOG_SIGMA2):
        upward = VariationalDropoutLinear(
            in_size, out_size * 4, nobias=nobias,
            initialW=initialW, initial_bias=initial_bias,
            p_threshold=p_threshold, loga_threshold=loga_threshold,
            initial_log_sigma2=initial_log_sigma2)
        lateral = VariationalDropoutLinear(
            in_size, out_size * 4, nobias=True,
            initialW=initialW,
            p_threshold=p_threshold, loga_threshold=loga_threshold,
            initial_log_sigma2=initial_log_sigma2)
        super(VariationalDropoutLSTM, self).__init__(
            upward=upward, lateral=lateral)
        self.in_size = in_size
        self.out_size = out_size
        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def set_state(self, c, h):
        self.h = h
        self.c = c

    def __call__(self, x):
        """Stateful LSTM call
        """
        memory_efficiency = configuration.config.user_memory_efficiency

        if memory_efficiency > 2:
            lstm_in = F.forget(self.upward, x)
        else:
            lstm_in = self.upward(x)
        if self.h is not None:
            if memory_efficiency > 2:
                lstm_in += F.forget(self.lateral, x)
            else:
                lstm_in += self.lateral(x)
        if self.c is None:
            self.c = self.xp.zeros((x.shape[0], self.out_size)).astype('f')

        if memory_efficiency > 1:
            self.c, self.h = F.forget(F.lstm, self.c, lstm_in)
        else:
            self.c, self.h = F.lstm(self.c, lstm_in)
        return self.h


def get_vd_link(link,
                p_threshold=P_THRESHOLD, loga_threshold=LOGA_THRESHOLD,
                initial_log_sigma2=INITIAL_LOG_SIGMA2):
    if link._cpu:
        gpu = -1
    else:
        gpu = link._device_id
        link.to_cpu()
    initialW = link.W.data
    initial_bias = getattr(link, 'b', None)
    if initial_bias is not None:
        initial_bias = initial_bias.data
    if type(link) == L.Linear:
        out_size, in_size = link.W.shape
        new_link = VariationalDropoutLinear(
            in_size=in_size, out_size=out_size, nobias=False,
            p_threshold=p_threshold, loga_threshold=loga_threshold,
            initial_log_sigma2=initial_log_sigma2)
    elif type(link) == L.Convolution2D:
        out_channels, in_channels = link.W.shape[:2]
        ksize = link.ksize
        stride = link.stride
        pad = link.pad
        new_link = VariationalDropoutConvolution2D(
            in_channels=in_channels, out_channels=out_channels,
            ksize=ksize, stride=stride, pad=pad,
            nobias=None, initialW=None, initial_bias=None,
            p_threshold=p_threshold, loga_threshold=loga_threshold,
            initial_log_sigma2=initial_log_sigma2)
    else:
        NotImplementedError()
    new_link.W.data[:] = numpy.array(initialW).astype('f')
    assert(numpy.any(link.W.data == new_link.W.data))
    if initial_bias is not None:
        new_link.b.data[:] = numpy.array(initial_bias).astype('f')
        assert(numpy.any(link.b.data == new_link.b.data))
    if gpu >= 0:
        new_link.to_gpu(gpu)
    return new_link


def to_variational_dropout_link(parent, name, link, path_name=''):
    raw_name = name.lstrip('/')
    print(raw_name)
    if isinstance(link, chainer.Chain):
        for child_name, child_link in sorted(
                link.namedlinks(skipself=True), key=lambda x: x[0]):
            to_variational_dropout_link(link, child_name, child_link,
                                        path_name=raw_name + '/')
    elif not '/' in raw_name:
        if not getattr(link, 'is_variational_dropout', False) and \
                type(link) in [L.Linear, L.Convolution2D]:
            new_link = get_vd_link(link.copy())
            delattr(parent, raw_name)
            parent.add_link(raw_name, new_link)
            print(' Replace link {} with a variant using variational dropout.'
                  .format(path_name + raw_name))

        else:
            print('  Retain link {}.'.format(path_name + raw_name))


class VariationalDropoutChain(chainer.link.Chain):

    def __init__(self, warm_up=0.0001, **kwargs):
        super(VariationalDropoutChain, self).__init__(**kwargs)
        self.warm_up = warm_up
        if self.warm_up:
            self.kl_coef = 0.
        else:
            self.kl_coef = 1.

    def calc_loss(self, x, t, add_kl=True, split_loss=False, calc_stats=True):
        train = configuration.config.train
        memory_efficiency = configuration.config.user_memory_efficiency

        self.y = self(x)
        if memory_efficiency > 0:
            self.class_loss = F.forget(F.softmax_cross_entropy, self.y, t)
        else:
            self.class_loss = F.softmax_cross_entropy(self.y, t)

        ignore = False
        if train and self.xp.isnan(self.class_loss.data):
            self.class_loss = chainer.Variable(
                self.xp.array(0.).astype('f').sum())
            ignore = True
        else:
            reporter.report({'class': self.class_loss.data}, self)

        if add_kl:
            a_regf = sum(
                calculate_kl(
                    link.W, link.loga_threshold,
                    log_sigma2=link.log_sigma2, log_alpha=None,
                    eps=1e-8, thresholds=(-8., 8.))
                for link in self.links()
                if getattr(link, 'is_variational_dropout', False))
            self.kl_loss = a_regf * self.kl_coef

            if train and self.xp.isnan(self.kl_loss.data):
                self.kl_loss = chainer.Variable(
                    self.xp.array(0.).astype('f').sum())
                ignore = True
            else:
                reporter.report({'kl': self.kl_loss.data}, self)
            self.kl_coef = min(self.kl_coef + self.warm_up, 1.)
            reporter.report({'kl_coef': self.kl_coef}, self)

            self.loss = self.class_loss + self.kl_loss
        else:
            self.loss = self.class_loss

        if not ignore:
            reporter.report({'loss': self.loss.data}, self)

        self.accuracy = F.accuracy(self.y.data, t).data
        reporter.report({'accuracy': self.accuracy}, self)

        if calc_stats:
            stats = calculate_stats(self)
            reporter.report({'mean_p': stats['mean_p']}, self)
            reporter.report({'sparsity': stats['sparsity']}, self)
            reporter.report({'W/Wnz': stats['W/Wnz']}, self)

        if split_loss:
            return self.class_loss, self.kl_loss
        else:
            return self.loss

    def to_cpu_sparse(self):
        self.to_cpu()
        n_total_old_params = 0
        n_total_new_params = 0
        if self.xp is not numpy:
            warnings.warn('SparseLinearForwardCPU link is made for'
                          ' inference usage. Please to_cpu()'
                          ' before inference.')
        print('Sparsifying fully-connected linear layer in the model...')
        for name, link in sorted(
                self.namedlinks(skipself=True), key=lambda x: x[0]):
            raw_name = name.lstrip('/')
            n_old_params = sum(p.size for p in link.params())

            if getattr(link, 'is_variational_dropout_linear', False):
                old = link.copy()
                delattr(self, raw_name)
                self.add_link(raw_name, old.get_sparse_cpu_model())
                n_new_params = getattr(self, raw_name).sparse_W.size
                if hasattr(getattr(self, raw_name), 'sparse_b'):
                    n_new_params += getattr(self, raw_name).sparse_b.size
                print(' Sparsified link {}.'.format(raw_name) +
                      '\t# of params: {} -> {} ({:.3f}%)'.format(
                          n_old_params, n_new_params,
                          (n_new_params * 1. / n_old_params * 100)))
                n_total_old_params += n_old_params
                n_total_new_params += n_new_params
            elif not isinstance(link, chainer.Chain):
                print('  Retain link {}.\t# of params: {}'.format(
                    raw_name, n_old_params))
                n_new_params = n_old_params
                n_total_old_params += n_old_params
                n_total_new_params += n_new_params
        print(' total # of params: {} -> {} ({:.3f}%)'.format(
            n_total_old_params, n_total_new_params,
            (n_total_new_params * 1. / n_total_old_params * 100)))

    def to_variational_dropout(self):
        """Make myself to use variational dropout

        Linear -> VariationalDropoutLinear
        Convolution2D -> VariationalDropoutConvolution2D

        """
        print('Make {} to use variational dropout.'.format(
            self.__class__.__mro__[2].__name__))
        for name, link in sorted(
                self.namedlinks(skipself=True), key=lambda x: x[0]):
            to_variational_dropout_link(self, name, link)


class VGG16VD(VariationalDropoutChain, vgg.VGG):

    def __init__(self, class_labels=10, warm_up=0.0001):
        super(VGG16VD, self).__init__(
            warm_up=warm_up, class_labels=class_labels)

