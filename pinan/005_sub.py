
import pandas as pd
data = pd.read_csv('data/store/6_20/13/submission.csv')
test = pd.read_excel('/mnt/2TB/jane96/pingan/base/public_test.xlsx')
train = pd.read_excel('/mnt/2TB/jane96/pingan/base/train.xlsx')['label']
y = train.values.tolist()
yLable = pd.Series(y).drop_duplicates().values

for index in range(1):
    sub = pd.read_csv('data/store/6_20/13/sub_0.csv').apply(lambda x : yLable[x])

result = test[['id','catgory']].values.tolist()
all = []
count = 0
sub = sub.values.tolist()

for index in range(len(result)):
    if(result[index][1] == 0):
        all.append(result[index][0] + '\t' + sub[count][0])
        count += 1




pd.Series(all).to_csv('data/store/6_20/13/submissions.csv',index=False,header=None)
