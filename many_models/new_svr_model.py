from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data_path = 'D:\GisProjects\Ward-Topics-Landuse-POI_Summary_by_GSS_Code.csv'
data = pd.read_csv(data_path,header=0)
# X = data.drop(columns = ['OID_','GSS_CODE','COUNT','COUNT_ldatopic','COUNT_ldatopic0','COUNT_ldatopic1','COUNT_ldatopic2',
#                          'COUNT_ldatopic3','COUNT_ldatopic4','COUNT_ldatopic5','COUNT_ldatopic6','COUNT_ldatopic7',
#                          'COUNT_topictime1','COUNT_topictime2','COUNT_topictime3','COUNT_topictime4','COUNT_topictime5',
#                          'COUNT_topictime6','COUNT_topseason0','COUNT_topseason1','COUNT_toptour0','COUNT_toptour1',
#                          'OBJECTID','GSS_CODE_1','COUNT_1'])
#toptime
X = data.drop(columns = ['OID_','GSS_CODE','COUNT','COUNT_ldatopic','COUNT_ldatopic0','COUNT_ldatopic1','COUNT_ldatopic2',
                         'COUNT_ldatopic3','COUNT_ldatopic4','COUNT_ldatopic5','COUNT_ldatopic6','COUNT_ldatopic7',
                         'COUNT_topictime5','COUNT_topseason0','COUNT_topseason1','COUNT_toptour0','COUNT_toptour1',
                         'OBJECTID','GSS_CODE_1','COUNT_1'])
#ldatopic3
# X = data.drop(columns = ['OID_','GSS_CODE','COUNT','COUNT_ldatopic','COUNT_ldatopic3',
#                          'COUNT_topictime1','COUNT_topictime2','COUNT_topictime3','COUNT_topictime4','COUNT_topictime5',
#                          'COUNT_topictime6','COUNT_topseason0','COUNT_topseason1','COUNT_toptour0','COUNT_toptour1',
#                          'OBJECTID','GSS_CODE_1','COUNT_1'])

#X = data.drop(columns =['OID_','GSS_CODE','COUNT','COUNT_ldatopic','COUNT_ldatopic3','OBJECTID','GSS_CODE_1','COUNT_1'])
Y = data['COUNT_topictime5']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
Svrmodel = SVR(kernel='linear',C=1.0, epsilon=0.2)
Svrmodel.fit(X_train,Y_train)
importance = Svrmodel.coef_
importance = [i for arr in importance for i in arr]
col = data.columns.tolist()
#toptime
to_remove = ['OID_','GSS_CODE','COUNT','COUNT_ldatopic','COUNT_ldatopic0','COUNT_ldatopic1','COUNT_ldatopic2',
             'COUNT_ldatopic3','COUNT_ldatopic4','COUNT_ldatopic5','COUNT_ldatopic6','COUNT_ldatopic7',
             'COUNT_topictime5','COUNT_topseason0','COUNT_topseason1','COUNT_toptour0','COUNT_toptour1',
             'OBJECTID','GSS_CODE_1','COUNT_1']
# to_remove = ['OID_','GSS_CODE','COUNT','COUNT_ldatopic','COUNT_ldatopic0','COUNT_ldatopic1','COUNT_ldatopic2',
#              'COUNT_ldatopic3','COUNT_ldatopic4','COUNT_ldatopic5','COUNT_ldatopic6','COUNT_ldatopic7',
#              'COUNT_topictime1','COUNT_topictime2','COUNT_topictime3','COUNT_topictime4','COUNT_topictime5',
#              'COUNT_topictime6','COUNT_topseason0','COUNT_topseason1','COUNT_toptour0','COUNT_toptour1',
#              'OBJECTID','GSS_CODE_1','COUNT_1']
#ldatopic3
# to_remove = ['OID_','GSS_CODE','COUNT','COUNT_ldatopic','COUNT_ldatopic3',
#              'COUNT_topictime1','COUNT_topictime2','COUNT_topictime3','COUNT_topictime4','COUNT_topictime5',
#              'COUNT_topictime6','COUNT_topseason0','COUNT_topseason1','COUNT_toptour0','COUNT_toptour1',
#              'OBJECTID','GSS_CODE_1','COUNT_1']
#to_remove = ['OID_','GSS_CODE','COUNT','COUNT_ldatopic','COUNT_ldatopic3','OBJECTID','GSS_CODE_1','COUNT_1']
col = [x for x in col if x not in to_remove]
f_importance = [np.round(x,4) for x in importance]

F2 = pd.Series(f_importance,index=col)
F2 = F2.sort_values(ascending=True)
print(F2)
Rsv = round(Svrmodel.score(X_test,Y_test)*100,2)
print(Rsv)
F2.to_csv('COUNT_topictime5.csv')