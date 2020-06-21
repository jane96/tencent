import pandas as pd

result = pd.DataFrame()
result['sid'] = pd.read_csv('/media/yza/f/py/tencent/data/base/test1.csv')['sid']
result['label'] = [0 for x in range(150000)]
for index in range(5):
    sub = pd.read_csv('/media/yza/f/py/tencent/data/store/6_20/10/sub_{}.csv'.format(index))
    result['label'] += sub['label']
# for index in range(5):
#     sub = pd.read_csv('/media/yza/f/py/tencent/data/store/6_20/1/sub_{}.csv'.format(index))
#     result['label'] += sub['label']
# for index in range(5):
#     sub = pd.read_csv('/media/yza/f/py/tencent/data/store/6_20/11/sub_{}.csv'.format(index))
#     result['label'] += sub['label']
result['label'] = result['label'].apply(lambda x : int(x/5 + 0.5))
result.to_csv('/media/yza/f/py/tencent/data/store/6_20/10/sub_test_1.csv')