
import pandas as pd



print('ok')


sub = pd.DataFrame()
age = pd.read_csv('/home/jane96/tencent/sub/submission.csv')
sub['user_id'] = age['user_id']
sub['predicted_age'] = [0 for x in range(1000000)]
sub['predicted_gender'] = [0 for x in range(1000000)]

gender = pd.read_pickle('/mnt/2TB/jane96/w2v/base/genderd_embedding_epoch_3_5flod.pkl')
age = pd.read_pickle('/mnt/2TB/jane96/w2v/base/age_embedding_epoch_3_5_epoch.pkl')
allage =[[0 for x in range(10)]for y in range(1000000)]
allgender =[[0 for x in range(2)]for y in range(1000000)]
for index in range(5):
    allgender += gender[index][0]
    allage += age[index]
sub['predicted_gender'] = allgender
for index in range(1,7):
    gender = pd.read_csv('/mnt/2TB/jane96/tencent/store/6_16/back/score_{}.csv'.format(index))
    # age = pd.read_csv('/mnt/2TB/jane96/tencent/store/6_16/back/score_{}.csv'.format(index))
    # gender['0'] = gender['0'].apply(lambda x : 1 if  x >= 0.5 else 2)
    # gender['user_id'] = age['user_id']
    # threshold = gender['1'].quantile(0.668416)
    # gender['gender'] = np.array([2 if x >= threshold else 1 for x in gender['1'].values.tolist()])
    sub['predicted_gender'] +=  gender['0']
sub['predicted_gender'] = sub['predicted_gender'].apply(lambda x : (x / 11))
sub['predicted_age'] = allage.argmax(1) + 1

sub['predicted_gender'] = sub['predicted_gender'].apply(lambda x : 1 if x >= 0.5 else 2)

# sub['predicted_age'] = age['predicted_age']
#age['predicted_gender'] = gender['predicted_gender']
sub.to_csv('/mnt/2TB/jane96/tencent/store/6_16/3_sub_test.csv',index=False)