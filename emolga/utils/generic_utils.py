from __future__ import absolute_import
from matplotlib.ticker import FuncFormatter
import numpy as np
import time
import sys
import six
import matplotlib.pyplot as plt
import matplotlib

def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return identifier


def make_tuple(*args):
    return args

def printv(v, prefix=''):
    if type(v) == dict:
        if 'name' in v:
            print(prefix + '#' + v['name'])
            del v['name']
        prefix += '...'
        for nk, nv in v.items():
            if type(nv) in [dict, list]:
                print(prefix + nk + ':')
                printv(nv, prefix)
            else:
                print(prefix + nk + ':' + str(nv))
    elif type(v) == list:
        prefix += '...'
        for i, nv in enumerate(v):
            print(prefix + '#' + str(i))
            printv(nv, prefix)
    else:
        prefix += '...'
        print(prefix + str(v))


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            return X[start]
        else:
            return X[start:stop]


class Progbar(object):
    def __init__(self, target, logger, width=30, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

        self.logger = logger

    def update(self, current, values=[]):
        '''
        @param current: index of current step
        @param values: list of tuples (name, value_for_last_step).
        The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('.'*(prog_width-1))
                if current < self.target:
                    bar += '(-w-)'
                else:
                    bar += '(-v-)!!'
            bar += ('~' * (self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)

            # info = ''
            info = bar
            if current < self.target:
                info += ' - Run-time: %ds - ETA: %ds' % (now - self.start, eta)
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if k == 'perplexity' or k == 'PPL':
                    info += ' - %s: %.4f' % (k, np.exp(self.sum_values[k][0] / max(1, self.sum_values[k][1])))
                else:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            # sys.stdout.write(info)
            # sys.stdout.flush()

            self.logger.info(info)

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                # sys.stdout.write(info + "\n")
                self.logger.info(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)

    def clear(self):
        self.sum_values = {}
        self.unique_values = []
        self.total_width = 0
        self.seen_so_far = 0


def print_sample(idx2word, idx):
    def cut_eol(words):
        for i, word in enumerate(words):
            if words[i] == '<eol>':
                return words[:i + 1]
        raise Exception("No end-of-line found")

    return cut_eol(map(lambda w_idx : idx2word[w_idx], idx))


def visualize_(subplots, data, w=None, h=None, name=None,
               display='on', size=10, text=None, normal=True,
               grid=False):
    fig, ax = subplots
    if data.ndim == 1:
        if w and h:
            # vector visualization
            assert w * h == np.prod(data.shape)
            data = data.reshape((w, h))
        else:
            L = data.shape[0]
            w = int(np.sqrt(L))
            while L % w > 0:
                w -= 1
            h = L / w
            assert w * h == np.prod(data.shape)
            data = data.reshape((w, h))
    else:
        w = data.shape[0]
        h = data.shape[1]

    if not size:
        size = 30 / np.sqrt(w * h)

    print(data.shape)

    major_ticks = np.arange(0, h, 1)
    ax.set_xticks(major_ticks)
    ax.set_xlim(0, h)
    major_ticks = np.arange(0, w, 1)
    ax.set_ylim(w, -1)
    ax.set_yticks(major_ticks)
    ax.set_aspect('equal')
    if grid:
        pass
        ax.grid(which='both')
        # ax.axis('equal')
    if normal:
        cax = ax.imshow(data, cmap=plt.cm.pink, interpolation='nearest',
                        vmax=1.0, vmin=0.0, aspect='auto')
    else:
        cax = ax.imshow(data, cmap=plt.cm.bone, interpolation='nearest', aspect='auto')

    if name:
        ax.set_title(name)
    else:
        ax.set_title('sample.')
    import matplotlib.ticker as ticker

    # ax.xaxis.set_ticks(np.arange(0, h, 1.))
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    # ax.yaxis.set_ticks(np.arange(0, w, 1.))
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    # ax.set_xticks(np.linspace(0, 1, h))
    # ax.set_yticks(np.linspace(0, 1, w))
    # Move left and bottom spines outward by 10 points
    # ax.spines['left'].set_position(('outward', size))
    # ax.spines['bottom'].set_position(('outward', size))
    # # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')

    if text:
        ax.set_yticks(np.linspace(0, 1, 33) * size * 3.2)
        ax.set_yticklabels([text[s] for s in range(33)])
    # cbar = fig.colorbar(cax)

    if display == 'on':
        plt.show()
    else:
        return ax


def vis_Gaussian(subplot, mean, std, name=None, display='off', size=10):
    ax   = subplot
    data = np.random.normal(size=(2, 10000))
    data[0] = data[0] * std[0] + mean[0]
    data[1] = data[1] * std[1] + mean[1]

    ax.scatter(data[0].tolist(), data[1].tolist(), 'r.')
    if display == 'on':
        plt.show()
    else:
        return ax