import pandas as pd
train = pd.read_csv('../data/base/train.csv')
test = pd.read_csv('../data/base/test1.csv')
col = ['android_id', 'apptype', 'carrier', 'dev_height', 'dev_ppi','timestamp',
       'dev_width', 'media_id', 'ntt', 'package',
       'version', 'fea_hash', 'location', 'fea1_hash',
       'cus_type']
col = ['android_id', 'apptype', 'carrier', 'dev_height',
       'dev_ppi', 'dev_width', 'lan', 'media_id', 'ntt', 'osv',
       'package',  'timestamp', 'version', 'fea_hash', 'location',
       'fea1_hash', 'cus_type']
##process timestamp
data = pd.concat([train[col],test[col]],axis= 0)
time_max = data['timestamp'].max()
time_min = data['timestamp'].min()
time_gap = (time_max - time_min) / 24
time_bins = [time_min + time_gap * x for x in range(25)]
time_labels = [x for x in range(24)]
data['timestamp'] = pd.cut(data['timestamp'],time_bins,labels=time_labels,right=True,include_lowest=False)

###process osv
data['osv'] = data['osv'].apply(lambda x : x.split('_')[1] if str(x).startswith('Android_') else x)
data['osv'] = data['osv'].apply(lambda x : x.split(' ')[1] if str(x).startswith('Android ') else x)
data['osv'] = data['osv'].fillna('8.1.0')
osv_9 = ['9.0.0','9.0','9']
osv_8 = ['8.1.0','8.1']
osv_8_0 = ['8.0.0','8.0','8']
osv_7 = ['7.0.0','7.0','7']
osv_6 = ['6.0.0','6.0','6']
osv_5 = ['5.1.0','5.1']

data['osv'] = data['osv'].apply(lambda x : '9.0.0' if x in osv_9 else x)
data['osv'] = data['osv'].apply(lambda x : '8.1.0' if x in osv_8 else x)
data['osv'] = data['osv'].apply(lambda x : '8.0.0' if x in osv_8_0 else x)
data['osv'] = data['osv'].apply(lambda x : '7.0.0' if x in osv_7 else x)
data['osv'] = data['osv'].apply(lambda x : '6.0.0' if x in osv_6 else x)
data['osv'] = data['osv'].apply(lambda x : '5.1.0' if x in osv_5 else x)

###process lan
# data['lan'] = data['lan'].fillna('zh-cn')
lan_zh = ['zh-CN','zh','cn','zh_CN','zh-cn','ZH','CN']
lan_tw = ['tw','zh-TW','TW']
data['lan'] = data['lan'].apply(lambda  x : 'zh-cn' if x in lan_zh else x)
data['lan'] = data['lan'].apply(lambda  x : 'tw' if x in lan_tw else x)

###process carrier
# data['carrier'] = data['carrier'].apply(lambda x : 46000 if x == -1 or x == 0 else int(x))

### process ntt
# data['ntt'] = data['ntt'].fillna(2)
# data['ntt'] = data['ntt'].apply(lambda x : 2 if x == 0 else x)

data['all'] = ''
prefix = ['a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','x','y','z']
count  = 0
for co in col:
       data['all'] +=  prefix[count] + data[co].astype(str) + ' '
       count += 1
data['all'].to_csv('../data/process/one_17_process_fill.csv',index=False)


