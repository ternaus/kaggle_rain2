#!/usr/bin/env python
'''
Data that we see for rain2 competition is messy. It requires some aggregation and cleaning
'''

import graphlab as gl
import numpy as np
from scipy.stats import skew

train = gl.SFrame('../data/train.csv')
test = gl.SFrame('../data/test.csv')

for column in train.column_names():
  train = train.fillna(column, 0)

for column in test.column_names():
  test = test.fillna(column, 0)

f = {'device_mean': {'device_max': max,
                     'device_min': min}
              }

f = {}
for column in ['radardist_km', 
               'Ref', 
               'Ref_5x5_10th', 
               'Ref_5x5_50th', 
               'Ref_5x5_90th', 
               'RefComposite', 
               'RefComposite_5x5_10th', 
               'RefComposite_5x5_50th', 
               'RefComposite_5x5_90th', 
               'RhoHV', 
               'RhoHV_5x5_10th', 
               'RhoHV_5x5_50th', 
               'RhoHV_5x5_90th', 
               'Zdr', 
               'Zdr_5x5_10th', 
               'Zdr_5x5_50th', 
               'Zdr_5x5_90th', 
               'Kdp', 
               'Kdp_5x5_10th', 
               'Kdp_5x5_50th', 
               'Kdp_5x5_90th']:
    f[column + '_mean'] =  gl.aggregate.AVG(column)
    f[column + '_std'] =  gl.aggregate.STD(column)
    f[column + '_var'] =  gl.aggregate.VAR(column)
    
    
f['Expected'] = gl.aggregate.AVG('Expected')

train_grouped = train.groupby("Id", f)

train_grouped = train_grouped[train_grouped['Ref_std'] != 0]

f = {}
for column in ['radardist_km', 
               'Ref', 
               'Ref_5x5_10th', 
               'Ref_5x5_50th', 
               'Ref_5x5_90th', 
               'RefComposite', 
               'RefComposite_5x5_10th', 
               'RefComposite_5x5_50th', 
               'RefComposite_5x5_90th', 
               'RhoHV', 
               'RhoHV_5x5_10th', 
               'RhoHV_5x5_50th', 
               'RhoHV_5x5_90th', 
               'Zdr', 
               'Zdr_5x5_10th', 
               'Zdr_5x5_50th', 
               'Zdr_5x5_90th', 
               'Kdp', 
               'Kdp_5x5_10th', 
               'Kdp_5x5_50th', 
               'Kdp_5x5_90th']:
    f[column + '_mean'] =  gl.aggregate.AVG(column)
    f[column + '_std'] =  gl.aggregate.STD(column)
    f[column + '_var'] =  gl.aggregate.VAR(column)

test_grouped = test.groupby("Id", f)

train_grouped.save('../data/train_grouped1.csv')
test_grouped.save('../data/test_grouped1.csv')