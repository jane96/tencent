import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

level = 104343
test = pd.read_excel('/mnt/2TB/jane96/pingan/base/public_test.xlsx')
train = pd.read_excel('/mnt/2TB/jane96/pingan/base/train.xlsx')

tr = train.values.tolist()
question = []
answer = []
qu = ''
# for index in range(len(tr)):
#     category = tr[index][1]
#     if(category == 1):
#         qu = tr[index][3]
#     else:
#         question.append(qu)
#         answer.append(tr[index][3])
# print(len(question))
# te = test.values.tolist()
# train_len = 50652
# for index in range(len(te)):
#     category = te[index][1]
#     if(category == 1):
#         qu = te[index][3]
#     else:
#         question.append(qu)
#         answer.append(te[index][3])
# pd.Series(answer).to_csv('/mnt/2TB/jane96/pingan/w2v/answer.csv')
# pd.Series(question).to_csv('/mnt/2TB/jane96/pingan/w2v/question.csv')


# y = train['label']
# train = train.drop('label',axis=1)
# Counter(y)
# ros = RandomOverSampler(random_state=0)
# sample_x,sample_y = ros.fit_resample(train,y)
# print(Counter(sample_y))




print('ok')




# df = pd.concat([train,test],axis=0)
# word = df['word'].apply(lambda  x : x.replace('[','').replace(']','').replace('\'','').replace(',',' '))
# word.to_csv('/mnt/2TB/jane96/pingan/w2v/allword.csv',index=False)


answer = pd.read_csv('/mnt/2TB/jane96/pingan/w2v/answer.csv')['0']
question = pd.read_csv('/mnt/2TB/jane96/pingan/w2v/question.csv')['0']
answer = answer.apply(lambda  x : x.replace('[','').replace(']','').replace('\'','').replace(',',' '))
question = question.apply(lambda  x : x.replace('[','').replace(']','').replace('\'','').replace(',',' '))
(answer).to_csv('/mnt/2TB/jane96/pingan/w2v/answer.csv')
(question).to_csv('/mnt/2TB/jane96/pingan/w2v/question.csv')
print('ok')