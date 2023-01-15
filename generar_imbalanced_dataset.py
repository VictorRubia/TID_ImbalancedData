import numpy as np

# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from matplotlib.colors import ListedColormap
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour  
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier

 
from pylab import rcParams
 
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
 
from collections import Counter

# Needed for generating classification, regression and clustering datasets
import sklearn.datasets as dt
from imblearn.over_sampling import ADASYN
from collections import Counter

# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def mostrar_resultados(y_test, pred_y):
    print (classification_report(y_test, pred_y))

# Define the seed so that results can be reproduced
seed = 11
rand_state = 11

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange","green"])

fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(16,5))
plt_ind_list = np.arange(3)+131

x,y = dt.make_classification(n_samples=10000,n_classes=2,weights= [0.1,0.9],
                                class_sep = 1.2,n_features=5,
                                n_informative=3,n_redundant=1,
                                n_clusters_per_class = 1,random_state=rand_state)
unos = 0
print("Hay de y ", len(y))
print("Hay de x ", len(x))
ceros = 0
for i in range(0, len(y)):
    if(y[i] == 1 ):
        unos = unos + 1
    if(y[i] == 0 ):
        ceros = ceros + 1

print("Hay de la clase 0 ", ceros, " y de la clase 1 ", unos)
pca = PCA(n_components=2)

# fit the PCA model and transform the data
X_reduced = pca.fit_transform(x)
print("Len x", len(x), " y de dimensión ", len(x[0]), " y de X_reduce ", len(X_reduced), " y de dimensión ", len(X_reduced[0]))


# use the first two principal components to form the new dataset
X_pca = X_reduced[:, :2]

# get the length of the new dimensionality
new_dimensionality_length = X_pca.shape[1]
print(new_dimensionality_length)
print("Len x", len(x), " y de X_pca ", len(X_pca))

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

print("X_train ", len(X_train), " X-test ", len(X_test))

unos = 0
ceros = 0
for i in range(0, len(y_train)):
    if(y_train[i] == 1 ):
        unos = unos + 1
    if(y_train[i] == 0 ):
        ceros = ceros + 1
print("Hay de la clase minoritaria ", ceros, " y de la clase mayoritaria ", unos)

cmap = ListedColormap(['#31b093', '#e3743a'])
# print(f"BEFORE: {X_pca.shape[1]} AFTER ")
my_scatter_plot = plt.scatter(X_pca[:,0], X_pca[:,1],c=y,vmin=min(y),vmax=max(y),s=1,cmap=cmap)

# plt.suptitle('make_classification() With Different class_sep Values',fontsize=20)
plt.show()
# print(f"BEFORE: {X_pca.shape[1]} AFTER ")
my_scatter_plot = plt.scatter(X_pca[:,0], X_pca[:,1],c=y,vmin=min(y),vmax=max(y),s=1,cmap=cmap)

# plt.suptitle('make_classification() With Different class_sep Values',fontsize=20)
plt.show()
# |L|   |S|
# 6780
# 980

# ada = ADASYN(random_state=rand_state, sampling_strategy=0.5)
# X_res, y_res = ada.fit_resample(X_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_res))

# ros = RandomOverSampler(random_state=rand_state, sampling_strategy=0.5)
# X_res, y_res = ros.fit_resample(X_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_res))

# sm = SMOTE(random_state=rand_state, sampling_strategy=0.5)
# X_res, y_res = sm.fit_resample(X_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_res))

# rus = RandomUnderSampler(random_state=rand_state, sampling_strategy=0.5)
# X_res, y_res = rus.fit_resample(X_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_res))

# nm = NearMiss(version=1, sampling_strategy=0.5)
# X_res, y_res = nm.fit_resample(X_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_res))

# cnn = CondensedNearestNeighbour(random_state=rand_state)  
# X_res, y_res = cnn.fit_resample(X_train, y_train)  
# print('Resampled dataset shape %s' % Counter(y_res))

# tl = TomekLinks()
# X_res, y_res = tl.fit_resample(X_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_res))

smt = SMOTETomek(random_state=rand_state, sampling_strategy=0.5)
X_res, y_res = smt.fit_resample(X_train, y_train)

# enn = EditedNearestNeighbours()
# X_res, y_res = enn.fit_resample(X_train, y_train)

# sme = SMOTEENN(random_state=rand_state, sampling_strategy=0.5)
# X_res, y_res = sme.fit_resample(X_train, y_train)

# eec = EasyEnsembleClassifier(random_state=rand_state)
# eec.fit(X_train, y_train)

# pred_y = eec.predict(X_test)

# bbc = BalancedBaggingClassifier(random_state=rand_state)
# bbc.fit(X_train, y_train)

# pred_y = bbc.predict(X_test)


# print('Resampled dataset shape %s' % Counter(y_res))

# my_scatter_plot = plt.scatter(X_res[:,0], X_res[:,1],c=y_res,vmin=min(y_res),vmax=max(y_res),s=1,cmap=color_map_discrete)

# plt.suptitle('make_classification() With Different class_sep Values',fontsize=20)
# plt.show()
lr2 = LogisticRegression(fit_intercept=True, penalty='l2', tol=1e-5, C=0.8, solver='lbfgs', max_iter=75,warm_start=True)
model = lr2.fit(X_res, y_res)
pred_y = model.predict(X_res)
mostrar_resultados(y_res, pred_y)
print(f"AUC-ROC: {roc_auc_score(y_res, lr2.predict_proba(X_res)[:, 1])}")


x_min, x_max = X_res[:, 0].min() - 0.5, X_res[:, 0].max() + 0.5
y_min, y_max = X_res[:, 1].min() - 0.5, X_res[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = lr2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap = ListedColormap(['#31b093', '#e3743a'])

plt.scatter(X_res[:, 0], X_res[:, 1], c=y_res, cmap=cmap, s=1)

# Plot the decision boundary of the classifier
plt.contour(xx, yy, Z, colors='black', linewidths=1)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# plt.suptitle('SMOTE y ENN',fontsize=20)

plt.show()
