# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:14:06 2020

@author: ayanca
"""

# Shapiro-Wilk Test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

# interpret
alpha = 0.05

def shapiro_normality_test(data):
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))    
    
    if p > alpha:
        return('Sample looks Gaussian (fail to reject H0)')
    else:
        return('Sample does not look Gaussian (reject H0)')
        
def d_agostino_k2_normality_test(data):
    stat,p = normaltest(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))    
    
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
        
def anderson_darling_normality_test(data):
    result = anderson(data)
    print('Statistic: %.3f' % result.statistic)   
    
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))