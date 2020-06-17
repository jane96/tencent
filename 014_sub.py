import pandas as pd
import numpy as np

sub = pd.DataFrame()
age = pd.read_csv('/home/jane96/tencent/sub/submission.csv')
sub['user_id'] = age['user_id']
sub['predicted_age'] = [0 for x in range(1000000)]
sub['predicted_gender'] = [0 for x in range(1000000)]
for index in range(1,2):
    gender = pd.read_csv('/mnt/2TB/jane96/tencent/store/6_16/age_1_{}.csv'.format(index))
    # gender['0'] = gender['0'].apply(lambda x : 1 if  x >= 0.5 else 2)
    # gender['user_id'] = age['user_id']
    # threshold = gender['1'].quantile(0.668416)
    # gender['gender'] = np.array([2 if x >= threshold else 1 for x in gender['1'].values.tolist()])
    sub['predicted_gender'] +=  gender['predicted_age']
sub['predicted_gender'] = sub['predicted_gender'].apply(lambda x : int(x / 1 + 0.5))

# sub['predicted_age'] = age['predicted_age']
#age['predicted_gender'] = gender['predicted_gender']
sub.to_csv('/mnt/2TB/jane96/tencent/store/6_16/1_sub_test.csv',index=False)
print('ok')
print('dsds')