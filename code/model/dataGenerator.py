###############################################################################
###                            dataGenerator.py                             ###
###############################################################################
import numpy as np
import matplotlib.pyplot as plt

class dataGenerator(object):
    def __init__(self, gb_data, org_data=None):
        gb_data.loc[gb_data['base'] > 3000, 'base'] = 3000
        self.gb_data = gb_data.loc[gb_data['gb'] < 30,:]
        self.org_data = org_data
        X = np.array(self.gb_data.iloc[:, 3:])
        self.X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        self.y = np.log(self.gb_data['gb'] + 1)
        self.weight = np.array(self.gb_data['base'])
        self.log_weight = np.log2(self.weight - min(self.weight) + 2)

    def split_train_test(self, test_size, random_state, weight='base'):
        np.random.seed(random_state)
        test_idx = np.random.choice(self.y.shape[0], int(self.y.shape[0] * test_size), replace=False)
        test_idx = np.array([i in test_idx for i in range(self.y.shape[0])])
        if weight == 'base':
            resp = np.array([self.y, self.weight / min(self.weight)]).T
        elif weight == 'log_base':
            resp = np.array([self.y, self.log_weight]).T
        else:
            raise ValueError('Unrecognizable weight: {}'.format(weight))
        self.X_train, self.X_test = self.X[~test_idx,:], self.X[test_idx,:]
        self.y_train, self.y_test = resp[~test_idx], resp[test_idx]
        self.weight_train, self.weight_test = self.weight[~test_idx], self.weight[test_idx]
        return self.X_train, self.X_test, self.y_train, self.y_test, self.weight_train, self.weight_test

    ### -----------------------   help function  ------------------------- ###
    @staticmethod
    def truncate(x, lower=None, upper=None):
        res = x
        if lower is None and upper is None:
            return res
        if lower is not None:
            lower_bound = np.percentile(x, lower)
            res[res < lower] = lower_bound
        if upper is not None:
            upper_bound = np.percentile(x, upper)
            res[res > upper] = upper_bound
        return res
