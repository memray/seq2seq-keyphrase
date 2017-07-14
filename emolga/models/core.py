import json

import numpy

from keyphrase import config

__author__ = 'jiataogu'
import theano
import logging
import deepdish as dd

from emolga.dataset.build_dataset import serialize_to_file, deserialize_from_file, serialize_to_file_json
from emolga.utils.theano_utils import floatX

logger = logging.getLogger(__name__)


class Model(object):
    def __init__(self):
        self.layers  = []
        self.params  = []
        self.monitor = {}
        self.watchlist = []

    def _add(self, layer):
        if layer:
            self.layers.append(layer)
            self.params += layer.params

    def _monitoring(self):
        # add monitoring variables
        for l in self.layers:
            for v in l.monitor:
                name = v + '@' + l.name
                print(name)
                self.monitor[name] = l.monitor[v]

    def compile_monitoring(self, inputs, updates=None):
        logger.info('compile monitoring')
        for i, v in enumerate(self.monitor):
            self.watchlist.append(v)
            logger.info('monitoring [{0}]: {1}'.format(i, v))

        self.watch = theano.function(inputs,
                                     [self.monitor[v] for v in self.watchlist],
                                     updates=updates
                                     )
        logger.info('done.')

    def set_weights(self, weights):
        if hasattr(self, 'save_parm'):
            params = self.params + self.save_parm
        else:
            params = self.params

        for p, w in zip(params, weights):
            # print(p.name)
            if p.eval().shape != w.shape:
                raise Exception("Layer shape %s not compatible with weight shape %s." % (p.eval().shape, w.shape))
            p.set_value(floatX(w))

    def get_weights(self):
        weights = []
        for p in self.params:
            weights.append(p.get_value())

        if hasattr(self, 'save_parm'):
            for v in self.save_parm:
                weights.append(v.get_value())

        return weights

    def set_name(self, name):
        for i in range(len(self.params)):
            if self.params[i].name is None:
                self.params[i].name = '%s_p%d' % (name, i)
            else:
                self.params[i].name = name + '@' + self.params[i].name
        self.name = name

    def save(self, filename):
        if hasattr(self, 'save_parm'):
            params = self.params + self.save_parm
        else:
            params = self.params
        ps = 'save: <\n'
        for p in params:
            ps += '{0}: {1}\n'.format(p.name, p.eval().shape)
        ps += '> to ... {}'.format(filename)
        # logger.info(ps)

        # hdf5 module seems works abnormal !!
        # dd.io.save(filename, self.get_weights())
        serialize_to_file(self.get_weights(), filename)

    def load(self, filename):
        logger.info('load the weights.')

        # hdf5 module seems works abnormal !!
        # weights = dd.io.load(filename)
        weights = deserialize_from_file(filename)
        # print(len(weights))
        self.set_weights(weights)

    def save_weight_json(self, filename):
        '''
        Write weights into file for manually check (not useful, too many parameters in (-1,1))
        :param filename:
        :return:
        '''
        print('Save weights into file: %s' % filename)
        with open(filename, 'w') as f:
            for p in self.params:
                str = json.dumps(p.get_value().tolist())+'\n'
                f.write(str)

    def load_weight_json(self, filename):
        '''
        Write weights into file for manually check (not useful, too many parameters in (-1,1))
        :param filename:
        :return:
        '''
        count_1 = 0
        count_5 = 0
        count_10 = 0
        total = 0
        max = 0.
        with open(filename, 'r') as f:
            for line in f:
                list = numpy.array(json.loads(line))
                list = list.ravel()
                for e in list:
                    total += 1
                    if abs(e) > 1:
                        count_1 += 1
                    if abs(e) > 5:
                        count_5 += 1
                    if abs(e) > 10:
                        count_10 += 1
                        print(e)
                    if abs(e) > max:
                        max = abs(e)
                        print('new max = %f' % e)
        print('total = %d' % total)
        print('count < 1/5/10 = %d / %d / %d' % (count_1, count_5, count_10))
        print('max = %f' % max)

if __name__ == '__main__':
    config = config.setup_keyphrase_all()  # setup_keyphrase_inspec
    agent = Model()
    agent.load_weight_json(config['weight_json'])