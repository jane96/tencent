import pandas as pd

result = pd.DataFrame()
result['sid'] = pd.read_csv('/mnt/2TB/jane96/track/data/test1.csv')['sid']
result['label'] = [0 for x in range(150000)]
for index in range(1):
    sub = pd.read_csv('/mnt/2TB/jane96/track/store/6_15/9/sub_{}.csv'.format(index))
    result['label'] += sub['label']

result['label'] = result['label'].apply(lambda x : int(x/1 + 0.5))
result.to_csv('/mnt/2TB/jane96/track/store/6_15/2/subAll_4.csv')