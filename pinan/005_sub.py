
import pandas as pd
# test = pd.read_excel('data/base/public_test.xlsx')
train = pd.read_excel('data/base/train.xlsx')
y = train.values.tolist()
yLable = pd.Series(y).drop_duplicates().values

for index in range(1):
    sub = pd.read_csv('data/store/6_20/13/sub_0.csv').apply(lambda x : yLable[x])

result = test[['id','catgory']]
result['label'] = sub
result = result[result['catgory']==0]
result = result['id'] + '\t' +  result['label']
result.to_csv('data/store/6_20/13/submission.csv',index=False,header=None)
