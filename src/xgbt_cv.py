from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
xgbt to predict how much rain we will get
'''
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
import numpy as np
import math
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_absolute_error

print 'read train'
train = pd.read_csv('../data/train_grouped.csv')
test = pd.read_csv('../data/test_grouped.csv')
# print 'read test'
# test = pd.read_csv('../data/test.csv')

train = train[~train['Ref_mean'].isnull()]

y = train['Expected']
X = train.drop(['Id', 'Expected'], 1)

print X.shape

X_test = test.drop('Id', 1)


print 'filling missing values'
for column in X.columns:
  a = X[column].mean()
  X[column] = X[column].fillna(a)
  X_test[column] = X_test[column].fillna(a)

params = {
  'objective': 'reg:linear',
  # 'objective': 'count:poisson',
  # 'eta': 0.005,
  # 'min_child_weight': 6,
  # 'subsample': 0.7,
  # 'colsabsample_bytree': 0.7,
  # 'scal_pos_weight': 1,
  'silent': 1,
  # 'max_depth': 9
}

num_rounds = 2000
random_state = 42
print X.shape
offset = int(0.2 * X.shape[0])
test_size = 0.2

ind = 2
if ind == 1:
  n_iter = 5
  rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=test_size, random_state=random_state)

  result = []
  for scale_pos_weight in [1]:
    for min_child_weight in [3]:
      for eta in [0.1]:
        for colsample_bytree in [0.5, 0.7, 0.9]:
          for max_depth in [7, 8]:
            for subsample in [0.7, 0.9, 1]:
              for gamma in [1]:
                params['min_child_weight'] = min_child_weight
                params['eta'] = eta
                params['colsample_bytree'] = colsample_bytree
                params['max_depth'] = max_depth
                params['subsample'] = subsample
                params['gamma'] = gamma
                params['scale_pos_weight'] = scale_pos_weight

                params_new = list(params.items())
                score = []
                # score_truncated_up = []
                # score_truncated_down = []
                score_truncated_both = []
                # score_truncated_both_round = []
                # score_truncated_both_int = []

                for train_index, test_index in rs:

                  X_train = X.values[train_index]
                  X_test = X.values[test_index]
                  y_train = y.values[train_index]
                  y_test = y.values[test_index]

                  xgtest = xgb.DMatrix(X_test)

                  xgtrain = xgb.DMatrix(X_train[offset:, :], label=y_train[offset:])
                  xgval = xgb.DMatrix(X_train[:offset, :], label=y_train[:offset])

                  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                  model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                  preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

                  # X_train = X_train[::-1, :]
                  # labels = y_train[::-1]

                  # xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
                  # xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

                  # watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                  # model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                  # preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

                  # preds = 0.5 * preds1 + 0.5 * preds2
                  preds = preds1
                  tp = mean_absolute_error(y_test, preds)
                  score += [tp]
                  print tp

                sc = math.ceil(100000 * np.mean(score)) / 100000
                sc_std = math.ceil(100000 * np.std(score)) / 100000
                result += [(sc,
                            sc_std,
                            min_child_weight,
                            eta,
                            colsample_bytree,
                            max_depth,
                            subsample,
                            gamma,
                            n_iter,
                            params['objective'],
                            test_size,
                            scale_pos_weight)]
                print result

    result.sort()

    print
    print 'result'
    print result


elif ind == 2:
  xgtrain = xgb.DMatrix(X.values[offset:, :], label=y.values[offset:])
  xgval = xgb.DMatrix(X.values[:offset, :], label=y.values[:offset])
  xgtest = xgb.DMatrix(X_test.values)

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  params = {
  'objective': 'reg:linear',
    # 'objective': 'count:poisson',
  'eta': 0.1,
  'min_child_weight': 1,
  'subsample': 0.7,
  'colsample_bytree': 0.5,
  # 'scal_pos_weight': 1,
  'silent': 1,
  'max_depth': 12,
  'gamma': 0
  }    
  params_new = list(params.items())
  model1 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
  prediction_test_1 = model1.predict(xgtest, ntree_limit=model1.best_iteration)

  X_train = X.values[::-1, :]
  labels = y.values[::-1]

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model2 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

  prediction_test_2 = model2.predict(xgtest, ntree_limit=model2.best_iteration)


  prediction_test = 0.5 * prediction_test_1 + 0.5 * prediction_test_2
  submission = pd.DataFrame()
  submission['Id'] = test['Id']
  submission['Expected'] = prediction_test
  submission.to_csv("predictions/xgbt.csv", index=False)

