# Data Science Competition Snippet

將資料科學競賽常用資料整理方法整理。

## EDA & Features Engineering 

### Download Google Drive Files without using Google Drive Mount.
```python
!pip install gdown
def Import_from_drive(id, output):
  !gdown --id {id} -O {output}
# token = token copy from google drive share 
Import_from_drive(token, 'data.7z')
```

### Histogram for each column of data.

```python
import pandas as pd
def df_hist(df):
  fig, axes = plt.subplots(len(df.columns)//10, 10, figsize=(36, 48))
  i = 0
  for triaxis in axes:
      for axis in triaxis:
          df.hist(column = df.columns[i], ax=axis, range=[min(df[df.columns[i]]), max(df[df.columns[i]])])
          i+=1
```

### Remove highly correlated columns 
```python
def corr_remove_col(df, th=0.8):

  df = pd.DataFrame(df)
  corr_matrix = df.corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
  to_drop = [column for column in upper.columns if any(upper[column] > th)]
  print("移除相關性高於於", th, "的欄位：", to_drop)
  return to_drop

df.drop(columns=drop_col)
```

### Remove outliers (abs(Z-score) >3)
```python
from scipy import stats
def outlier_remove(df):
  abs_z_scores = np.abs(stats.zscore(df))
  filtered_entries = (abs_z_scores < 3.5).all(axis=1)
  new_df = df[filtered_entries]
  print("移除："'{:.2%}樣本'.format(1-(new_df.shape[0]/df.shape[0])))
  return new_df
```

### Z-score Scaler(using Sci-kit Learn API)
```python
from sklearn.preprocessing import StandardScaler
def zscore(df):
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df)
  return df_scaled
```

## Modeling

### Train-Test split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_df, train_label, test_size=0.3, random_state=420)

```

### Multi-Label Confusion Metrix
```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report


def conf_matrix_report(y_true, y_pred):  
  conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
  #
  # Print the confusion matrix using Matplotlib
  #
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
  for i in range(conf_matrix.shape[0]):
      for j in range(conf_matrix.shape[1]):
          ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
  
  plt.xlabel('Predictions', fontsize=18)
  plt.ylabel('Actuals', fontsize=18)
  plt.title('Confusion Matrix', fontsize=18)
  plt.show()

  print(classification_report(y_true, y_pred, digits=3))
```

### Hypertuning XGBoost Model using [hyperopt](https://github.com/hyperopt/hyperopt)

```python
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

space={'max_depth': hp.quniform("max_depth", 5, 20, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'eval_metric':'merror','objective':'multi:softmax', 'num_class':3,
        'seed': 420
    }
def objective(space):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)
    num_round = 100
    watchlist = [(dval, 'eval'), (dtrain, 'train')]
    param = {'max_depth':int(space['max_depth']), 
             'eta':0.7,
             'objective':'multi:softmax', 
             'num_class':3,
             'colsample_bytree':int(space['colsample_bytree']), 
             'eval_metric':'merror', 
             'min_child_weight': int(space['min_child_weight'])
             }
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=10, verbose_eval=False)

    pred = bst.predict(dval)
    accuracy = f1_score(y_test, pred, average='macro')
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 20,
                        trials = trials)
```
