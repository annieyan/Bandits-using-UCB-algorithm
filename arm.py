'''
implement a Bernoulli distributed arm
'''
import numpy as np
import random
# from scipy.stats import bernoulli

class arm(object):
    def __init__(self,p):
        self.p=p
        self.expectation = p

    '''
    pull an arm according to Bernoulli, return 1 or 0
    '''
    def draw_sample(self):
        # random.random() Return the next random floating point number in the range [0.0, 1.0).
        return float(random.random()<self.p)
        