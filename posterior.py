'''
implement Thompson Sampling
'''

import numpy as np 
import random
from scipy.stats import beta

'''
Beta distribution as the posteriors of Bernoulli
for each time t, for each arm i do,
sample hat_mu_i(t) from Beta(S_t(t)+1,F_i(t)+1)
'''
class posterior(object):
    '''
    input  S_t(t)+1,F_i(t)+1
    ''' 
    def __init__(self,a=1,b=1):
        self.a = a
        self.b=b

 
    def sample(self):
        # generate one sample
        return beta.rvs(self.a, self.b, size=1)

    def get_var(self):
        return beta.var(self.a, self.b)
        