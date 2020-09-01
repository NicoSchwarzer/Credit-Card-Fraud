# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:08:42 2020

@author: Nico
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:00:12 2020

@author: Nico
"""

#######################
#######################
## Tech Labs Projekt ##
#######################
#######################



# Grundlegende Module laden

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv(r"C:\Users\Nico\Desktop\file1.csv", sep = "\,")

df1.shape
# Das sind 284807 Zeilen und 31 Spalten (Variablen) 
df1.head()


############################
## Vorbereitung der Daten ##
############################

# Inspizierung des Datensatzes
df1.info()

## Die abhängige Variable, "Class" ist vom Typ integer, der Rest sind float64

### Nans?
df1.isnull().values.any()
# "False", also keine nans, sher gut :D


## Für sehr starke Multikorrelationen checken
import seaborn as sns

df1_corr = df1.drop(["Time", "Class"], axis = 1).corr()
df1_corr = pd.DataFrame(df1_corr)

print(df1_corr)
# ziemlich unübersichtlich

sns.heatmap(df1_corr, annot = False)
## Es scheinen zunächst keine Korrelationen verdächtig zu sein

### Ausreißer
# Das ist eigentlich sehr schwierig zu beurteilen, da wir ja
# nicht wissen, für was die Variablen stehen

# Ausreißer mit Percentilen aussortieren

# Dabei lasse ich Class außenvor, da die 1-Werte ja praktisch Ausreißer
# sind, da es so wenige von ihnen gibt :D
df1_not_class = df1.drop("Class", axis = 1)

vars_list = list(df1_not_class.columns)

q_list = []    
q_list = list(q_list)

for var in vars_list:
    q = np.percentile(df1[var],99.9)
    q_list.append(q)
    
q_list

dfx = pd.DataFrame(q_list)
dfx["99%-percentile"] = dfx[0]
dfx["var"] = vars_list
dfx_999 = dfx.drop(0, axis = 1)
dfx_999

# Dieser Datensatz gibt die jeweiligen 99%- Perzentil an
# Erstellen eines Datensatzes ohne die Werte im obersten 1%-Percentil

df_no_outlier_999 = pd.DataFrame(df1_not_class)

for var in vars_list:
    q = np.percentile(df1[var],99.9)
    df_no_outlier_999 = df_no_outlier_999[df_no_outlier_999[var] < q]
        
df_no_outlier_999.shape
df1_not_class.shape
print( df1.shape[0] - df_no_outlier_999.shape[0])
rel_loss =  54589 / 284807
print(rel_loss)
# selbst bei einem Niveau von 99.9% würden 19% der Einträge gelöscht werden
# vielleicht deswegen erstmals keine Ausreißer löschen ?!


# Visuelle Inspektion für Ausreißer
vars_list = list(df1.columns)

for var in vars_list:
    print(var)
    plt.boxplot(df1[var])
# auch hieraus lässt sich per sé nichts schließen

# Daten als categorical codieren, da es sich ja nicht
# um eine kontinuierliche Variable handelt
Class_categorical = pd.Categorical(df1["Class"])
Class_categorical = pd.DataFrame(Class_categorical)

##### Standardisierter Datensatz für manche Algorithmen
# nicht für die Variable Class, da diese ja schon binär ist!

from sklearn.preprocessing import scale

df1_not_class = df1.drop("Class", axis = 1)

df_not_class_scaled = scale(df1_not_class)
df_not_class_scaled = pd.DataFrame(df_not_class_scaled)

##############################
######## Datenanalyse ########
##############################

## Performance - Bewertung:

# Unsere Daten haben folgende Eigenschaften:
# 1. die abhängige Variable ist bnär
# 2. es gibt DEUTLICH mehr 0 als 1 Werte
# Um die Performance unserer Algorithmen zu bewerten,
# braucht es also folgende Methoden:


def accuracy(AA):
    # Accuracy brauchen wir eigentlich eben nicht, aber ich code es dennoch :D
    a = (AA[0,0] + AA[1,1] ) / (AA[0,0] + AA[0,1] + AA[1,0] + AA[1,1])
    print(a)

def precision(AA):
    # dividing the true pos. by all pos.
    p = AA[0,0]  / (AA[0,0] + AA[0,1])
    print(p)
    
def recall(AA):    
    # dividing the true pos. by (true pos. + neg. false)
    r = AA[0,0] / (AA[0,0] + AA[1,0])
    print(r)
    
def scores(AA):
    print("accuracy: " + str(accuracy(AA)))
    print("precision: " + str(precision(AA)))
    print("recall: " + str(recall(AA)))
    
    
#######################
# K-nearest neighbors #
#######################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

y_knn1 = df1["Class"]
X_knn1 = df1.drop('Class', axis=1).values

X1_train, X1_test, y1_train, y1_test = train_test_split(X_knn1, y_knn1, test_size = 0.3, random_state = 21)

knn1 = KNeighborsClassifier(n_neighbors=6)

knn1.fit(X1_train, y1_train)

y1_pred = knn1.predict(X1_test)

knn1.score(X1_test, y1_test) 
# 0.9983029622087239
# aber nicht sehr aussagekräftig bei kategorischen Daten!

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

list_pred_1 = []

for a in y1_pred:
    if a < 0.5:
        list_pred_1.append(0)
    else:
        list_pred_1.append(1)
        
cm1 = confusion_matrix(y1_test, list_pred_1)
print(cm1)
scores(cm1)

# Noch einmal mit dem standardisierten Datensatz
df_not_class_scaled = pd.DataFrame(df_not_class_scaled)

y_knn2 = df1['Class']
X_knn2 = df_not_class_scaled

X2_train, X2_test, y2_train, y2_test = train_test_split(X_knn2, y_knn2, test_size = 0.3, random_state = 21)

knn2 = KNeighborsClassifier(n_neighbors=6)

knn2.fit(X2_train, y2_train)

y2_pred = knn2.predict(X2_test)

list_pred_2 = []

for a in y2_pred:
    if a < 0.5:
        list_pred_2.append(0)
    else:
        list_pred_2.append(1)
        
cm2 = confusion_matrix(y1_test, list_pred_2)
print(cm2)
scores(cm2)

## Für die optimale Anzahl an 'Neighbors' eine Abfrageschleife:

n = [2,3,4,5,6]

for a in n:
    knn2_n = KNeighborsClassifier(n_neighbors=a)
    knn2_n.fit(X2_train, y2_train)
    y2_pred = knn2_n.predict(X2_test)
    knn2_n.score(X2_test, y2_test) 
    print(confusion_matrix(y2_test, y2_pred))
    print(classification_report(y2_test, y2_pred))
    cm = confusion_matrix(y2_test, y2_pred)
    print(scores(cm))
    
# mit 5 "neighbors" gehts am besten, aber immer noch nicht gut!!
    
##########################
# Logistische regression #
##########################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

y_log1 = df1['Class']
X_log1 = df1.drop('Class', axis=1).values

X3_train, X3_test, y3_train, y3_test = train_test_split(X_log1, y_log1, test_size = 0.3, random_state = 21)

log1 = LogisticRegression()

log1.fit(X3_train, y3_train)

y3_pred = log1.predict(X3_test)

# score
log1.score(X3_test, y3_test)

cm4 = confusion_matrix(y3_test, y3_pred)
#[[85270    24]
 #[   62    87]] --> echt schlecht
scores(cm4)
 
#ROC - Kurve
from sklearn.metrics import roc_curve

y3_pred_prob = log1.predict_proba(X3_test) [:, 1]
fpr, tpr, tresholds = roc_curve(y3_test, y3_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

plt.savefig("plot_log1")

#saving the model
import pickle
filename = r'C:\Users\Nico\Documents\TechLabs.log1'
pickle.dump(log1, open(filename, 'wb'))

## nochmal mit dem standardisierten Datensatz
df_not_class_scaled = pd.DataFrame(df_not_class_scaled)

y_log2 = df1['Class']
X_log2 = df_not_class_scaled
X4_train, X4_test, y4_train, y4_test = train_test_split(X_log2, y_log2, test_size = 0.3, random_state = 21)

log2 = LogisticRegression()
log2.fit(X4_train, y4_train)

y4_pred = log2.predict(X4_test)

# score
log2.score(X4_test, y4_test)
# 0.9991573329588147

cm5 = print(confusion_matrix(y4_test, y4_pred))
#[[85283    11]
 #[   61    88]] --> schlecht, aber ein wenig besser als nicht standardisiert

scores(cm5)

#saving the model
import pickle
filename = r'C:\Users\Nico\Documents\TechLabs.log2'
pickle.dump(log2, open(filename, 'wb'))

# ROC - curve
y4_pred_prob = log2.predict_proba(X3_test) [:, 1]
fpr, tpr, tresholds = roc_curve(y4_test, y4_pred_proba)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

## noch einmal mit weniger Variablen
# siehe weiter unten :D
df2 = df1[["V4", "V7", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18", "V21", "V26", "V27"]]

y_log3 = df1['Class']
X_log3 = df2

X3a_train, X3a_test, y3a_train, y3a_test = train_test_split(X_log3, y_log3, test_size = 0.3, random_state = 21)

log3 = LogisticRegression()

log3.fit(X3a_train, y3a_train)

y3a_pred = log3.predict(X3a_test)

# score
log3.score(X3a_test, y3a_test)
# 0.9991105181231933 - not bad!

cm6 = confusion_matrix(y3a_test, y3a_pred)
#[[85284    10]
# [   66    83]] -- nicht wirklich besser!

scores(cm6)

# ROC - Kurve
from sklearn.metrics import roc_curve

y3a_pred_prob = log3.predict_proba(X3a_test) [:, 1]
fpr, tpr, tresholds = roc_curve(y3a_test, y3a_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

import pickle
filename = r'C:\Users\Nico\Documents\TechLabs.log3'
pickle.dump(log3, open(filename, 'wb'))

######################
### Random Forests ###
######################


from sklearn.ensemble import RandomForestClassifier

forest1 = RandomForestClassifier(n_estimators = 110,
                                max_depth = 10,
                                max_features = 4,
                                random_state = 42)

y_rf1 = df1['Class']
X_rf1 = df1.drop('Class', axis=1).values

X5_train, X5_test, y5_train, y5_test = train_test_split(X_rf1, y_rf1, test_size = 0.3, random_state = 21)

forest1.fit(X5_train, y5_train)

y5_pred = forest1.predict(X5_test)

list_pred_5 = []

for a in y5_pred:
    if a < 0.5:
        list_pred_5.append(0)
    else:
        list_pred_5.append(1)
        
cm7 = confusion_matrix(y5_test, list_pred_5)
print(cm7)
#[[85291     3]
# [   43   106]] - besser als zuvor
scores(cm7)

# Feature importances
imp = forest1.feature_importances_
print(imp)

importances = pd.DataFrame(vars_list)
importances["Vars"] = importances[0]
importances = importances.drop(0, axis = 1)
importances["Wichtigkeit"] = imp 
print(importances)


## sotzing by importance
sort = np.argsort(imp)[::-1]
x = range(len(imp))
labels = np.array(vars_list)
plt.bar(x, imp, tick_label = labels)
plt.xticks(rotation = 90)
plt.show()

plt.savefig("imp1")

# wichtige Variablen:V4, V7, V9, V10, V11, V12
# V14, V16, V17, V18, V21, V26, V27

df2 = df1[["V4", "V7", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18", "V21", "V26", "V27"]]
# alternativ: 
df2 = df1[["Time", "V3", "V4", "V7","V9","V10","V11","V12","V14","V16","V17","V18", "V21","V26","V27"]]

## hier wissen wir jetzt, welche features am 
# wichtigsten sind!!

# noch einmal den RFC anwenden

forest1 = RandomForestClassifier(n_estimators = 100,
                                max_depth = 8,
                                max_features = 4,
                                random_state = 42)

y_rf2 = df1['Class']
X_rf2 = df2

X6_train, X6_test, y6_train, y6_test = train_test_split(X_rf2, y_rf2, test_size = 0.3, random_state = 21)

forest1.fit(X6_train, y6_train)

y6_pred = forest1.predict(X6_test)

list_pred_6 = []

for a in y6_pred:
    if a < 0.5:
        list_pred_6.append(0)
    else:
        list_pred_6.append(1)
        
y6_pred = forest1.predict(X6_test)

cm8 = confusion_matrix(y6_test, list_pred_6)
print(cm8)
#[[85289     5]
#[   40   109]] - nicht wirklich besser als zuvor
scores(cm8)

#saving the model
import pickle
#filename = r'C:\Users\Nico\Documents\TechLabs.f1'
#pickle.dump(forest1, open(filename, 'wb'))

###  Mit den transformierten Daten ###

forest2 = RandomForestClassifier(n_estimators = 100,
                                max_depth = 8,
                                max_features = 4,
                                random_state = 42)

df_not_class_scaled = pd.DataFrame(df_not_class_scaled)

y_rf2 = df1['Class']
X_rf2 = df_not_class_scaled

X7_train, X7_test, y7_train, y7_test = train_test_split(X_rf2, y_rf2, test_size = 0.3, random_state = 21)

forest2.fit(X7_train, y7_train)

y7_pred = forest2.predict(X7_test)

list_pred_7 = []

for a in y7_pred:
    if a < 0.5:
        list_pred_7.append(0)
    else:
        list_pred_7.append(1)
        
cm9 = confusion_matrix(y7_test, list_pred_7)
print(cm9)
#  nicht  besser 
scores(cm9)

#saving the model
import pickle
#filename = r'C:\Users\Nico\Documents\TechLabs.f2'
#pickle.dump(forest2, open(filename, 'wb'))

### andere Parameterwerte ####

forest3 = RandomForestClassifier(n_estimators = 170,
                                max_depth = 14,
                                max_features = 4,
                                random_state = 42)

df_not_class_scaled = pd.DataFrame(df_not_class_scaled)

y_rf3 = df1['Class']
X_rf3 = df_not_class_scaled

X8_train, X8_test, y8_train, y8_test = train_test_split(X_rf3, y_rf3, test_size = 0.3, random_state = 21)

forest3.fit(X8_train, y8_train)

y8_pred = forest3.predict(X8_test)

list_pred_8 = []

for a in y8_pred:
    if a < 0.5:
        list_pred_8.append(0)
    else:
        list_pred_8.append(1)
        
y8_pred = forest3.predict(X8_test)

cm8 = confusion_matrix(y8_test, list_pred_8)
print(cm8)
# nicht gut!
scores(cm8)

#forest3.save(r'C:\Users\Nico\Documents\TechLabs.f3')

#saving the model
import pickle
#filename = r'C:\Users\Nico\Documents\TechLabs.f3'
#pickle.dump(forest3, open(filename, 'wb'))
 

#####################
## Neuronales Netz ##
#####################

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report


y_n2 = df1['Class']
X_n2 = df_not_class_scaled

X10_train, X10_test, y10_train, y10_test = train_test_split(X_n2, y_n2, test_size = 0.4, random_state = 21)

model10 = Sequential()
cols = X10_train.shape[1]

# first layer
model10.add(Dense(20, activation = "relu", input_shape = (cols, )))
# second layer
model10.add(Dense(20, activation = "relu"))
# output layer
model10.add(Dense(1))

model10.compile(optimizer = 'adam', loss = 'mean_squared_error')

early_stopping_monitor = EarlyStopping(patience = 20)

model10.fit(X10_test, y10_test, epochs = 100,
          callbacks = [early_stopping_monitor])


pred10 = model10.predict(X10_test)

from keras.models import load_model

#model10.save(r'C:\Users\Nico\Documents\TechLabs.m10')


model10.save('my_model_10')
model10 = load_model('my_model_10')


# wieder zu binären Variabblen zurück
list_pred_10 = []

for a in pred10:
    if a < 0.5:
        list_pred_10.append(0)
    else:
        list_pred_10.append(1)
        
print(list_pred_10)
list_pred_10a = pd.DataFrame(list_pred_10)

y10_test = pd.DataFrame(y10_test)

y10_test["pred"] =  list_pred_10
y10_test["abs_loss"] = y10_test["Class"] - y10_test["pred"] 
y10_test["abs_loss"] = abs(y10_test["abs_loss"])

np.mean(y10_test["abs_loss"])
# 0.0003247807729782397

# besser:
list_pred_10 = []

for a in y10_pred:
    if a < 0.5:
        list_pred_10.append(0)
    else:
        list_pred_10.append(1)

cm10 = confusion_matrix(y10_pred, list_pred_10)
print(cm10)
scores(cm10)

# geht noch besser!

## nun mit df2
df2 = df1[["Time", "V3", "V4", "V7","V9","V10","V11","V12","V14","V16","V17","V18", "V21","V26","V27"]]


y_n3 = df1['Class']
X_n3 = df2

X11_train, X11_test, y11_train, y11_test = train_test_split(X_n3, y_n3, test_size = 0.4, random_state = 21)

model11 = Sequential()
cols = X11_train.shape[1]

# first layer
model11.add(Dense(20, activation = "relu", input_shape = (cols, )))
# second layer
model11.add(Dense(20, activation = "relu"))
# third layer
model11.add(Dense(20, activation = "relu"))
# output layer
model11.add(Dense(1))


model11.compile(optimizer = 'adam', loss = 'mean_squared_error')

early_stopping_monitor = EarlyStopping(patience = 20)

model11.fit(X11_test, y11_test, epochs = 100,
          callbacks = [early_stopping_monitor])


pred11 = model11.predict(X11_test)


from keras.models import load_model

#model11.save(r'C:\Users\Nico\Documents\TechLabs.m11')

model11.save()

model11.save('my_model_11')
model11 = load_model('my_model_11')

# wieder zu binären Variabblen zurück
list_pred_11 = []

for a in pred11:
    if a < 0.5:
        list_pred_11.append(0)
    else:
        list_pred_11.append(1)
        
print(list_pred_11)
list_pred_11a = pd.DataFrame(list_pred_11)

y11_test = pd.DataFrame(y11_test)

y11_test["pred"] =  list_pred_11
y11_test["abs_loss"] = y11_test["Class"] - y11_test["pred"] 
y11_test["abs_loss"] = abs(y11_test["abs_loss"])

np.mean(y11_test["abs_loss"])
# 0.0016677931585369066
# deutlich mehr als bei der standardisiertem Datensatz

cm11 = confusion_matrix(y11_pred, list_pred_11)
print(cm11)
scores(cm11)


#############################
## Support vector machines ##
#############################

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 


y_svm1 = df1['Class']
X_svm1 = df1.drop('Class', axis=1).values

X13_train, X13_test, y13_train, y13_test = train_test_split(X_svm1, y_svm1, test_size = 0.4, random_state = 21)

## C of 1

model_svm1 = SVC(kernel = 'linear' , C=1)
model1_svm.fit(X13_train, y13_train)



# besser, da binär:
y13_pred = model1.predict(X13_test)

list_pred_13 = []

for a in y13_pred:
    if a < 0.5:
        list_pred_13.append(0)
    else:
        list_pred_13.append(1)

cm10 = confusion_matrix(y13_pred, list_pred_13)
print(cm10)
#[[113839      0]
 #[     0     84]]

scores(cm10)
# alles 1.0!

##### PEERFEKT!! Eine lineare SVM scheint also sehr gut zu passen!

from sklearn.externals import joblib
#filename = r'C:\Users\Nico\Documents\TechLabs.f3_1'
#joblib.dump(model_svm1, filename)

## einmal noch nur mit dem reduziertem Datensatz
df2 = df1[["Time", "V3", "V4", "V7","V9","V10","V11","V12","V14","V16","V17","V18", "V21","V26","V27"]]

y_svm2 = df1['Class']
X_svm2 = df2


X14_train, X14_test, y14_train, y14_test = train_test_split(X_svm2, y_svm2, test_size = 0.4, random_state = 21)


model_svm2 = SVC(kernel = 'linear' , C=1)
model_svm2.fit(X14_train, y14_train)


# besser, da binär:
y14_pred = model_svm2.predict(X14_test)

list_pred_14 = []

for a in y14_pred:
    if a < 0.5:
        list_pred_14.append(0)
    else:
        list_pred_14.append(1)

cm11 = confusion_matrix(y14_pred, list_pred_14)
print(cm11)
#[[113839      0]
 #[     0     84]]

scores(cm11)
# alles 1 :D

from sklearn.externals import joblib
#filename = r'C:\Users\Nico\Documents\TechLabs.f3_2'
#joblib.dump(model_svm2, filename)


## noch einmal mit dem standardisiertem Datensatz

y_svm3 = df1['Class']
X_svm3 = df_not_class_scaled

X15_train, X15_test, y15_train, y15_test = train_test_split(X_svm3, y_svm3, test_size = 0.4, random_state = 21)

model_svm3 = SVC(kernel = 'linear' , C=1)
model_svm3.fit(X15_train, y15_train)

# besser, da binär:
y15_pred = model_svm3.predict(X15_test)

list_pred_15 = []

for a in y15_pred:
    if a < 0.5:
        list_pred_15.append(0)
    else:
        list_pred_15.append(1)

cm12 = confusion_matrix(y15_pred, list_pred_15)
print(cm12)
#[[113752      0]
# [     0    171]]
# wie geil :D
scores(cm12)
# alles 1

from sklearn.externals import joblib
#filename = r'C:\Users\Nico\Documents\TechLabs.f3_3'
#joblib.dump(model_svm3, filename)