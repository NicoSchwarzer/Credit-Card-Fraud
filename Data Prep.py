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
    
    


