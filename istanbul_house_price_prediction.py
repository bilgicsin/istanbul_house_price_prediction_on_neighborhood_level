import pandas as pd
import sklearn as sk
import numpy as np
from sqlalchemy import create_engine
import pymysql
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#importing mysql table as dataframe
db_connection_str = 'mysql+pymysql://sinan:indeed87_@localhost/bitirme'
db_connection = create_engine(db_connection_str)
emlakdf = pd.read_sql('SELECT district,neighborhood,price,square,room FROM bitirme.hurriyetemlak_v4', con=db_connection)

#split dataframe for anomali detection based on  district
adalardf = emlakdf.loc[emlakdf['district'] == 'adalar']
adalardf = emlakdf.loc[emlakdf['district'] == 'adalar']
arnavutkoydf = emlakdf.loc[emlakdf['district'] == 'arnavutkoy']
atasehirdf = emlakdf.loc[emlakdf['district'] == 'ataşehir']
avcilardf = emlakdf.loc[emlakdf['district'] == 'avcılar']
bagcilardf = emlakdf.loc[emlakdf['district'] == 'bağcılar']
bahcelievlerdf = emlakdf.loc[emlakdf['district'] == 'bahcelievler']
bakirkoydf = emlakdf.loc[emlakdf['district'] == 'bakırkoy']
basaksehirdf = emlakdf.loc[emlakdf['district'] == 'başakşehir']
bayrampasadf = emlakdf.loc[emlakdf['district'] == 'bayrampaşa']
besiktasdf = emlakdf.loc[emlakdf['district'] == 'beşiktaş']
beykozdf = emlakdf.loc[emlakdf['district'] == 'beykoz']
beylikduzudf = emlakdf.loc[emlakdf['district'] == 'beylikduzu']
beyogludf = emlakdf.loc[emlakdf['district'] == 'beyoğlu']
buyukcekmecedf = emlakdf.loc[emlakdf['district'] == 'buyukcekmece']
catalcadf = emlakdf.loc[emlakdf['district'] == 'catalca']
cekmekoydf = emlakdf.loc[emlakdf['district'] == 'cekmekoy']
esenlerdf = emlakdf.loc[emlakdf['district'] == 'esenler']
esenyurtdf = emlakdf.loc[emlakdf['district'] == 'esenyurt']
eyupdf = emlakdf.loc[emlakdf['district'] == 'eyup']
fatihdf = emlakdf.loc[emlakdf['district'] == 'fatih']
gaziosmanpasadf = emlakdf.loc[emlakdf['district'] == 'gaziosmanpaşa']
gungorendf = emlakdf.loc[emlakdf['district'] == 'gungoren']
kadikoydf = emlakdf.loc[emlakdf['district'] == 'kadıkoy']
kagithanedf = emlakdf.loc[emlakdf['district'] == 'kağıthane']
kartaldf = emlakdf.loc[emlakdf['district'] == 'kartal']
kucukcekmecedf = emlakdf.loc[emlakdf['district'] == 'kucukcekmece']
maltepedf = emlakdf.loc[emlakdf['district'] == 'maltepe']
pendikdf = emlakdf.loc[emlakdf['district'] == 'pendik']
sancaktepedf = emlakdf.loc[emlakdf['district'] == 'sancaktepe']
sariyerdf = emlakdf.loc[emlakdf['district'] == 'sarıyer']
silivridf = emlakdf.loc[emlakdf['district'] == 'silivri']
sultanbeylidf = emlakdf.loc[emlakdf['district'] == 'sultanbeyli']
sultangazidf = emlakdf.loc[emlakdf['district'] == 'sultangazi']
siledf = emlakdf.loc[emlakdf['district'] == 'şile']
sislidf = emlakdf.loc[emlakdf['district'] == 'şişli']
tuzladf = emlakdf.loc[emlakdf['district'] == 'tuzla']
umraniyedf = emlakdf.loc[emlakdf['district'] == 'umraniye']
uskudardf = emlakdf.loc[emlakdf['district'] == 'uskudar']
zeytinburnudf = emlakdf.loc[emlakdf['district'] == 'zeytinburnu']

#seeing outliars for zeytinburnu

plt.scatter(zeytinburnudf.price, zeytinburnudf.square, color='blue', s=1, alpha=1)
plt.title("Price vs Square for Zeytinburnu", loc='left')
plt.xlabel('Price')
plt.ylabel('Square')
plt.show()

#detecting outliars with isolation forest

to_model_columns = zeytinburnudf.columns[2:4]
clf=IsolationForest(n_estimators=100, max_samples=50000, contamination=float(.1),max_features=0.50, bootstrap=False,
                    n_jobs=-1, random_state=42, verbose=0)
clf.fit(zeytinburnudf[to_model_columns])
pred = clf.predict(zeytinburnudf[to_model_columns])
zeytinburnudf['outlier']=pred
outliers=zeytinburnudf.loc[zeytinburnudf['outlier']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
#print(zeytinburnudf['outlier'].value_counts())

zeytinburnudfanom = zeytinburnudf.loc[zeytinburnudf['outlier'] == -1]
zeytinburnudfnorm = zeytinburnudf.loc[zeytinburnudf['outlier'] ==  1]
b1=plt.scatter(zeytinburnudfanom.price, zeytinburnudfanom.square, color='red', s=1, alpha=1)
b2=plt.scatter(zeytinburnudfnorm.price, zeytinburnudfnorm.square, color='blue', s=1, alpha=1)
plt.title("Price vs Square for Zeytinburnu", loc='left')
plt.xlabel('Price')
plt.ylabel('Square')
plt.legend((b1,b2),('outlier','norm value'),scatterpoints=1,loc='upper right',ncol=3,fontsize=8)
plt.show()

#filter outliers from df
print(zeytinburnudfnorm)

#doing same thing for other 38 district

to_model_columns=adalardf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(adalardf[to_model_columns])
pred=clf.predict(adalardf[to_model_columns])
adalardf['outlier']=pred
outliers=adalardf.loc[adalardf['outlier']==-1]
outlier_index=list(outliers.index)
adalardfnorm=adalardf.loc[adalardf['outlier']==1]

to_model_columns=arnavutkoydf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(arnavutkoydf[to_model_columns])
pred=clf.predict(arnavutkoydf[to_model_columns])
arnavutkoydf['outlier']=pred
outliers=arnavutkoydf.loc[arnavutkoydf['outlier']==-1]
outlier_index=list(outliers.index)
arnavutkoydfnorm=arnavutkoydf.loc[arnavutkoydf['outlier']==1]

to_model_columns=atasehirdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(atasehirdf[to_model_columns])
pred=clf.predict(atasehirdf[to_model_columns])
atasehirdf['outlier']=pred
outliers=atasehirdf.loc[atasehirdf['outlier']==-1]
outlier_index=list(outliers.index)
atasehirdfnorm=atasehirdf.loc[atasehirdf['outlier']==1]

to_model_columns=avcilardf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(avcilardf[to_model_columns])
pred=clf.predict(avcilardf[to_model_columns])
avcilardf['outlier']=pred
outliers=avcilardf.loc[avcilardf['outlier']==-1]
outlier_index=list(outliers.index)
avcilardfnorm=avcilardf.loc[avcilardf['outlier']==1]

to_model_columns=bagcilardf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(bagcilardf[to_model_columns])
pred=clf.predict(bagcilardf[to_model_columns])
bagcilardf['outlier']=pred
outliers=bagcilardf.loc[bagcilardf['outlier']==-1]
outlier_index=list(outliers.index)
bagcilardfnorm=bagcilardf.loc[bagcilardf['outlier']==1]

to_model_columns=bahcelievlerdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(bahcelievlerdf[to_model_columns])
pred=clf.predict(bahcelievlerdf[to_model_columns])
bahcelievlerdf['outlier']=pred
outliers=bahcelievlerdf.loc[bahcelievlerdf['outlier']==-1]
outlier_index=list(outliers.index)
bahcelievlerdfnorm=bahcelievlerdf.loc[bahcelievlerdf['outlier']==1]

to_model_columns=bakirkoydf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(bakirkoydf[to_model_columns])
pred=clf.predict(bakirkoydf[to_model_columns])
bakirkoydf['outlier']=pred
outliers=bakirkoydf.loc[bakirkoydf['outlier']==-1]
outlier_index=list(outliers.index)
bakirkoydfnorm=bakirkoydf.loc[bakirkoydf['outlier']==1]

to_model_columns=basaksehirdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(basaksehirdf[to_model_columns])
pred=clf.predict(basaksehirdf[to_model_columns])
basaksehirdf['outlier']=pred
outliers=basaksehirdf.loc[basaksehirdf['outlier']==-1]
outlier_index=list(outliers.index)
basaksehirdfnorm=basaksehirdf.loc[basaksehirdf['outlier']==1]

to_model_columns=bayrampasadf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(bayrampasadf[to_model_columns])
pred=clf.predict(bayrampasadf[to_model_columns])
bayrampasadf['outlier']=pred
outliers=bayrampasadf.loc[bayrampasadf['outlier']==-1]
outlier_index=list(outliers.index)
bayrampasadfnorm=bayrampasadf.loc[bayrampasadf['outlier']==1]

to_model_columns=besiktasdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(besiktasdf[to_model_columns])
pred=clf.predict(besiktasdf[to_model_columns])
besiktasdf['outlier']=pred
outliers=besiktasdf.loc[besiktasdf['outlier']==-1]
outlier_index=list(outliers.index)
besiktasdfnorm=besiktasdf.loc[besiktasdf['outlier']==1]

to_model_columns=beykozdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(beykozdf[to_model_columns])
pred=clf.predict(beykozdf[to_model_columns])
beykozdf['outlier']=pred
outliers=beykozdf.loc[beykozdf['outlier']==-1]
outlier_index=list(outliers.index)
beykozdfnorm=beykozdf.loc[beykozdf['outlier']==1]

to_model_columns=beylikduzudf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(beylikduzudf[to_model_columns])
pred=clf.predict(beylikduzudf[to_model_columns])
beylikduzudf['outlier']=pred
outliers=beylikduzudf.loc[beylikduzudf['outlier']==-1]
outlier_index=list(outliers.index)
beylikduzudfnorm=beylikduzudf.loc[beylikduzudf['outlier']==1]

to_model_columns=beyogludf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(beyogludf[to_model_columns])
pred=clf.predict(beyogludf[to_model_columns])
beyogludf['outlier']=pred
outliers=beyogludf.loc[beyogludf['outlier']==-1]
outlier_index=list(outliers.index)
beyogludfnorm=beyogludf.loc[beyogludf['outlier']==1]

to_model_columns=buyukcekmecedf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(buyukcekmecedf[to_model_columns])
pred=clf.predict(buyukcekmecedf[to_model_columns])
buyukcekmecedf['outlier']=pred
outliers=buyukcekmecedf.loc[buyukcekmecedf['outlier']==-1]
outlier_index=list(outliers.index)
buyukcekmecedfnorm=buyukcekmecedf.loc[buyukcekmecedf['outlier']==1]

to_model_columns=catalcadf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(catalcadf[to_model_columns])
pred=clf.predict(catalcadf[to_model_columns])
catalcadf['outlier']=pred
outliers=catalcadf.loc[catalcadf['outlier']==-1]
outlier_index=list(outliers.index)
catalcadfnorm=catalcadf.loc[catalcadf['outlier']==1]

to_model_columns=cekmekoydf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(cekmekoydf[to_model_columns])
pred=clf.predict(cekmekoydf[to_model_columns])
cekmekoydf['outlier']=pred
outliers=cekmekoydf.loc[cekmekoydf['outlier']==-1]
outlier_index=list(outliers.index)
cekmekoydfnorm=cekmekoydf.loc[cekmekoydf['outlier']==1]

to_model_columns=esenlerdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(esenlerdf[to_model_columns])
pred=clf.predict(esenlerdf[to_model_columns])
esenlerdf['outlier']=pred
outliers=esenlerdf.loc[esenlerdf['outlier']==-1]
outlier_index=list(outliers.index)
esenlerdfnorm=esenlerdf.loc[esenlerdf['outlier']==1]

to_model_columns=esenyurtdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(esenyurtdf[to_model_columns])
pred=clf.predict(esenyurtdf[to_model_columns])
esenyurtdf['outlier']=pred
outliers=esenyurtdf.loc[esenyurtdf['outlier']==-1]
outlier_index=list(outliers.index)
esenyurtdfnorm=esenyurtdf.loc[esenyurtdf['outlier']==1]

to_model_columns=eyupdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(eyupdf[to_model_columns])
pred=clf.predict(eyupdf[to_model_columns])
eyupdf['outlier']=pred
outliers=eyupdf.loc[eyupdf['outlier']==-1]
outlier_index=list(outliers.index)
eyupdfnorm=eyupdf.loc[eyupdf['outlier']==1]

to_model_columns=fatihdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(fatihdf[to_model_columns])
pred=clf.predict(fatihdf[to_model_columns])
fatihdf['outlier']=pred
outliers=fatihdf.loc[fatihdf['outlier']==-1]
outlier_index=list(outliers.index)
fatihdfnorm=fatihdf.loc[fatihdf['outlier']==1]

to_model_columns=gaziosmanpasadf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(gaziosmanpasadf[to_model_columns])
pred=clf.predict(gaziosmanpasadf[to_model_columns])
gaziosmanpasadf['outlier']=pred
outliers=gaziosmanpasadf.loc[gaziosmanpasadf['outlier']==-1]
outlier_index=list(outliers.index)
gaziosmanpasadfnorm=gaziosmanpasadf.loc[gaziosmanpasadf['outlier']==1]

to_model_columns=gungorendf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(gungorendf[to_model_columns])
pred=clf.predict(gungorendf[to_model_columns])
gungorendf['outlier']=pred
outliers=gungorendf.loc[gungorendf['outlier']==-1]
outlier_index=list(outliers.index)
gungorendfnorm=gungorendf.loc[gungorendf['outlier']==1]

to_model_columns=kadikoydf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(kadikoydf[to_model_columns])
pred=clf.predict(kadikoydf[to_model_columns])
kadikoydf['outlier']=pred
outliers=kadikoydf.loc[kadikoydf['outlier']==-1]
outlier_index=list(outliers.index)
kadikoydfnorm=kadikoydf.loc[kadikoydf['outlier']==1]

to_model_columns=kagithanedf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(kagithanedf[to_model_columns])
pred=clf.predict(kagithanedf[to_model_columns])
kagithanedf['outlier']=pred
outliers=kagithanedf.loc[kagithanedf['outlier']==-1]
outlier_index=list(outliers.index)
kagithanedfnorm=kagithanedf.loc[kagithanedf['outlier']==1]

to_model_columns=kartaldf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(kartaldf[to_model_columns])
pred=clf.predict(kartaldf[to_model_columns])
kartaldf['outlier']=pred
outliers=kartaldf.loc[kartaldf['outlier']==-1]
outlier_index=list(outliers.index)
kartaldfnorm=kartaldf.loc[kartaldf['outlier']==1]

to_model_columns=kucukcekmecedf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(kucukcekmecedf[to_model_columns])
pred=clf.predict(kucukcekmecedf[to_model_columns])
kucukcekmecedf['outlier']=pred
outliers=kucukcekmecedf.loc[kucukcekmecedf['outlier']==-1]
outlier_index=list(outliers.index)
kucukcekmecedfnorm=kucukcekmecedf.loc[kucukcekmecedf['outlier']==1]

to_model_columns=maltepedf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(maltepedf[to_model_columns])
pred=clf.predict(maltepedf[to_model_columns])
maltepedf['outlier']=pred
outliers=maltepedf.loc[maltepedf['outlier']==-1]
outlier_index=list(outliers.index)
maltepedfnorm=maltepedf.loc[maltepedf['outlier']==1]

to_model_columns=pendikdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(pendikdf[to_model_columns])
pred=clf.predict(pendikdf[to_model_columns])
pendikdf['outlier']=pred
outliers=pendikdf.loc[pendikdf['outlier']==-1]
outlier_index=list(outliers.index)
pendikdfnorm=pendikdf.loc[pendikdf['outlier']==1]

to_model_columns=sancaktepedf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(sancaktepedf[to_model_columns])
pred=clf.predict(sancaktepedf[to_model_columns])
sancaktepedf['outlier']=pred
outliers=sancaktepedf.loc[sancaktepedf['outlier']==-1]
outlier_index=list(outliers.index)
sancaktepedfnorm=sancaktepedf.loc[sancaktepedf['outlier']==1]

to_model_columns=sariyerdf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(sariyerdf[to_model_columns])
pred=clf.predict(sariyerdf[to_model_columns])
sariyerdf['outlier']=pred
outliers=sariyerdf.loc[sariyerdf['outlier']==-1]
outlier_index=list(outliers.index)
sariyerdfnorm=sariyerdf.loc[sariyerdf['outlier']==1]

to_model_columns=silivridf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(silivridf[to_model_columns])
pred=clf.predict(silivridf[to_model_columns])
silivridf['outlier']=pred
outliers=silivridf.loc[silivridf['outlier']==-1]
outlier_index=list(outliers.index)
silivridfnorm=silivridf.loc[silivridf['outlier']==1]

to_model_columns=sultanbeylidf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(sultanbeylidf[to_model_columns])
pred=clf.predict(sultanbeylidf[to_model_columns])
sultanbeylidf['outlier']=pred
outliers=sultanbeylidf.loc[sultanbeylidf['outlier']==-1]
outlier_index=list(outliers.index)
sultanbeylidfnorm=sultanbeylidf.loc[sultanbeylidf['outlier']==1]

to_model_columns=sultangazidf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(sultangazidf[to_model_columns])
pred=clf.predict(sultangazidf[to_model_columns])
sultangazidf['outlier']=pred
outliers=sultangazidf.loc[sultangazidf['outlier']==-1]
outlier_index=list(outliers.index)
sultangazidfnorm=sultangazidf.loc[sultangazidf['outlier']==1]

to_model_columns=siledf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(siledf[to_model_columns])
pred=clf.predict(siledf[to_model_columns])
siledf['outlier']=pred
outliers=siledf.loc[siledf['outlier']==-1]
outlier_index=list(outliers.index)
siledfnorm=siledf.loc[siledf['outlier']==1]

to_model_columns=sislidf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(sislidf[to_model_columns])
pred=clf.predict(sislidf[to_model_columns])
sislidf['outlier']=pred
outliers=sislidf.loc[sislidf['outlier']==-1]
outlier_index=list(outliers.index)
sislidfnorm=sislidf.loc[sislidf['outlier']==1]

to_model_columns=tuzladf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(tuzladf[to_model_columns])
pred=clf.predict(tuzladf[to_model_columns])
tuzladf['outlier']=pred
outliers=tuzladf.loc[tuzladf['outlier']==-1]
outlier_index=list(outliers.index)
tuzladfnorm=tuzladf.loc[tuzladf['outlier']==1]

to_model_columns=umraniyedf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(umraniyedf[to_model_columns])
pred=clf.predict(umraniyedf[to_model_columns])
umraniyedf['outlier']=pred
outliers=umraniyedf.loc[umraniyedf['outlier']==-1]
outlier_index=list(outliers.index)
umraniyedfnorm=umraniyedf.loc[umraniyedf['outlier']==1]

to_model_columns=uskudardf.columns[2:4]
clf=IsolationForest(n_estimators=100,max_samples=50000,contamination=float(.1),max_features=0.50,bootstrap=False,n_jobs=-1,random_state=42,verbose=0)
clf.fit(uskudardf[to_model_columns])
pred=clf.predict(uskudardf[to_model_columns])
uskudardf['outlier']=pred
outliers=uskudardf.loc[uskudardf['outlier']==-1]
outlier_index=list(outliers.index)
uskudardfnorm=uskudardf.loc[uskudardf['outlier']==1]

#union all frames to get all data without outliers

dataframes = [adalardfnorm,arnavutkoydfnorm,atasehirdfnorm,avcilardfnorm,bagcilardfnorm,bahcelievlerdfnorm,
              bakirkoydfnorm,basaksehirdfnorm,bayrampasadfnorm,besiktasdfnorm,beykozdfnorm,beylikduzudfnorm,
              beyogludfnorm,buyukcekmecedfnorm,catalcadfnorm,cekmekoydfnorm,esenlerdfnorm,esenyurtdfnorm,eyupdfnorm,
              fatihdfnorm,gaziosmanpasadfnorm,gungorendfnorm,kadikoydfnorm,kagithanedfnorm,kartaldfnorm,kucukcekmecedfnorm,
              maltepedfnorm,pendikdfnorm,sancaktepedfnorm,sariyerdfnorm,silivridfnorm,sultanbeylidfnorm,sultangazidfnorm,
              siledfnorm,sislidfnorm,tuzladfnorm,umraniyedfnorm,uskudardfnorm,zeytinburnudfnorm]
emlakdfnorm = pd.concat((dataframes))
emlakdfnorm = emlakdfnorm.drop(columns='outlier')


#applying K-means to find out possible groups

kmeans_model_columns=emlakdfnorm.columns[2:4]
kmeans = KMeans(n_clusters=750).fit(emlakdfnorm[kmeans_model_columns])
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

emlakdfnorm['pricepersquare'] = emlakdfnorm.price/emlakdfnorm.square
emlakdfnorm['clusters'] = labels

plt.scatter(emlakdfnorm['price'], emlakdfnorm['square'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

#appending mean cluster number as score to neighborhoods
scoredf = emlakdfnorm['clusters'].groupby(emlakdfnorm['neighborhood']).mean()
emlakdfnormsc = emlakdfnorm.merge(scoredf, on='neighborhood', how='left')
emlakdfnormscrn = emlakdfnormsc.rename(columns={"clusters_x":"cluster","clusters_y":"score"})

print(emlakdfnormscrn)

#appliying various algoritm for to predicting house prices
prediction_model_columns =['square','room','score']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(emlakdfnormscrn[prediction_model_columns], emlakdfnormscrn[['price']],test_size=0.2)

models = [LinearRegression(),
        RandomForestRegressor(n_estimators=100, max_features='sqrt'),
        KNeighborsRegressor(n_neighbors=6),
        SVR(kernel='linear'),
        LogisticRegression()
        ]
TestModels = pd.DataFrame()
tmp = {}
for model in models:
    # get model name
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
    # fit model on training dataset
    model.fit(Xtrain, Ytrain['price'])
    # predict prices for test dataset and calculate r^2
    tmp['R2_Price'] = r2_score(Ytest['price'], model.predict(Xtest))
    # write obtained data
    TestModels = TestModels.append([tmp])
TestModels.set_index('Model', inplace=True)

fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()

#RandomForestRegressor selected because it has highest r2 value

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(Xtrain, Ytrain['price'])
predictions = rf.predict(Xtest)
errors = abs(predictions - Ytest['price'])
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / Ytest['price'])
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(prediction_model_columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#plot data real value vs predicted value

c1=plt.scatter(Ytest['price'], Xtest['square'], color='blue', s=1, alpha=1)
c2=plt.scatter(predictions,Xtest['square'], color='red', s=1, alpha=1)
plt.title("Price vs Square for both actual value and predictions", loc='left')
plt.xlabel('Price')
plt.ylabel('Square')
plt.legend((c1,c2),('actual value','prediction'),scatterpoints=1,loc='upper right',ncol=3,fontsize=8)
plt.show()
