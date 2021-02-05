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



