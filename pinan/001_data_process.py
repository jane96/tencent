import pandas as pd
level = 104343
test = pd.read_excel('data/base/public_test.xlsx')
train = pd.read_excel('data/base/train.xlsx')
df = pd.concat([train,test],axis=0)
word = df['word'].apply(lambda  x : x.replace('[','').replace(']','').replace('\'','').replace(',',' '))
word.to_csv('data/w2v/allword.csv',index=False)



print('ok')