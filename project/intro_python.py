"""
Introduction to Scikit-Learn
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);

iris = sns.load_dataset('iris')
iris.head()

X_iris = iris.drop('species', axis=1)
X_iris.shape
y_iris = iris['species']
y_iris.shape

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model

X = x[:, np.newaxis]
model.fit(X, y)
model.coef_
model.intercept_

xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit);

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,test_size = 0.33,
                                                random_state=1)
from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)   
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);

from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')
    
X = digits.data
X.shape
y = digits.target
y.shape

from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)
y_model = model.predict(X)
from sklearn.metrics import accuracy_score
accuracy_score(y, y_model)

from sklearn.cross_validation import train_test_split
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(X, y, random_state=0,train_size=0.5)
# fit the model on one set of data
model.fit(X1, y1)
# evaluate the model on the second set of data
y2_model = model.predict(X2)
accuracy_score(y2, y2_model)

y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)

from sklearn.cross_validation import cross_val_score
cross_val_score(model, X, y, cv=5)

from sklearn.cross_validation import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut(len(X)))
scores
scores.mean()

#### Categorical Features
data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)
vec.get_feature_names()
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)
#### End Categorical Features

#### Text Features
sample = ['problem of evil','evil queen','horizon problem']
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample)
X
import pandas as pd
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
#### End Text Features

#### Imputation of Missing Data
from numpy import nan
X = np.array([[ nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   nan, 6  ],
              [ 8,   8,   1  ]])
y = np.array([14, 16, -1,  8, -5])
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean')
X2 = imp.fit_transform(X)
X2
#### End Imputation of Missing Data

##### Naive Bayes Classification
%matplotlib inline
%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

## Gaussian Naive Bayes
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y);
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);

yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)
## End Gaussian Naive Bayes

## Multinomial Naive Bayes
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names

categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(train.data, train.target)
labels = model.predict(test.data)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
## End Multinomial Naive Bayes
##### End Naive Bayes Classification

###### Linear Regression
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit);
print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)

rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])
model.fit(X, y)
print(model.intercept_)
print(model.coef_)

from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])

from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit);

from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)
        
gauss_model = make_pipeline(GaussianFeatures(20),LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10);

model = make_pipeline(GaussianFeatures(30),LinearRegression())
model.fit(x[:, np.newaxis], y)
plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5);

def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    
    if title:
        ax[0].set_title(title)

    ax[1].plot(model.steps[0][1].centers_,
               model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location',
              ylabel='coefficient',
              xlim=(0, 10))
    
model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)

## Ridge regression
from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge Regression')

from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso Regression')
###### End Linear Regression

##### Support Vector Machines
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5)

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)

from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)
model.support_vectors_

from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none');
            
X, y = make_blobs(n_samples=100, centers=2,random_state=0, cluster_std=1.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

X, y = make_blobs(n_samples=100, centers=2,random_state=0, cluster_std=0.8)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
    axi.set_title('C = {0:.1f}'.format(C), size=14)
##### End Support Vector Machines
    
##### Decision Tree and Random Forest
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

digits = load_digits()
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,random_state=0)
X, y = make_blobs(n_samples=300, centers=4,random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');

from sklearn.tree import DecisionTreeClassifier
tree_gini = DecisionTreeClassifier(criterion='gini',max_depth=3)
tree_gini.fit(Xtrain, ytrain)
y_pred_gini = tree_gini.predict(Xtest)
accuracy_score(ytest,y_pred)
mat_gini = confusion_matrix(ytest, y_pred_gini)

tree_entropy = DecisionTreeClassifier(criterion='entropy',max_depth=3)
tree_entropy.fit(Xtrain, ytrain)
y_pred_entropy = tree_entropy.predict(Xtest)
mat_entropy = confusion_matrix(ytest, y_pred_entropy)

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
visualize_classifier(DecisionTreeClassifier(), X, y)

from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,random_state=1)
bag.fit(X, y)
visualize_classifier(bag, X, y)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_classifier(model, X, y);

rng = np.random.RandomState(42)
x = 10 * rng.rand(200)
def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))

    return slow_oscillation + fast_oscillation + noise

y = model(x)
plt.errorbar(x, y, 0.3, fmt='o')

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)
xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])
ytrue = model(xfit, sigma=0)
plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
plt.plot(xfit, yfit, '-r');
plt.plot(xfit, ytrue, '-k', alpha=0.5)

from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')    
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
from sklearn import metrics
print(metrics.classification_report(ypred, ytest))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
##### End Decision Tree and Random Forest

##### Principal Component Analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print(pca.components_)
print(pca.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()

pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

##### Manifold Learning
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def make_hello(N=1000, rseed=42):
    # Make a plot with "HELLO" text; save as PNG
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)
    
    # Open this PNG and draw random points from it
    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]

X = make_hello(1000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal')

def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)  
X2 = rotate(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal')

from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
D.shape

plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar()

D2 = pairwise_distances(X2)
np.allclose(D, D2)

from sklearn.manifold import MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal')

def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])
    
X3 = random_projection(X, 3)
X3.shape

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2],**colorize)
ax.view_init(azim=70, elev=50)

model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal')

def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T
XS = make_hello_s_curve(X)

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2],**colorize)

from sklearn.manifold import MDS
model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal')

from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified',eigen_solver='dense')
out = model.fit_transform(XS)
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15)

##### k-Means Clustering
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
y_kmeans
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')

centers, labels = find_clusters(X, 4, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')

labels = KMeans(6, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')

from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)

labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
    
from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

from sklearn.manifold import TSNE

# Project the data: this step will take several seconds
tsne = TSNE(n_components=2, init='random', random_state=0)
digits_proj = tsne.fit_transform(digits.data)

# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

# Permute the labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# Compute the accuracy
accuracy_score(digits.target, labels)

##### Gaussian Mixture Models
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting
# Plot the data with K Means Labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)

from sklearn.mixture import GMM
gmm = GMM(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

probs = gmm.predict_proba(X)
print(probs[:5].round(3))

size = 50 * probs.max(1) ** 2  # square emphasizes differences
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
        
gmm = GMM(n_components=4, random_state=42)
plot_gmm(gmm, X)

gmm = GMM(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched)

##### Kernel Density Estimation
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

x = make_data(1000)

hist = plt.hist(x, bins=30, normed=True)
density, bins, patches = hist
widths = bins[1:] - bins[:-1]
(density * widths).sum()

x = make_data(20)
bins = np.linspace(-5, 10, 10)
fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                       sharex=True, sharey=True,
                       subplot_kw={'xlim':(-4, 9),
                                   'ylim':(-0.02, 0.3)})
fig.subplots_adjust(wspace=0.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, normed=True)
    ax[i].plot(x, np.full_like(x, -0.01), '|k',
               markeredgewidth=1)

fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -0.1), '|k',
        markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1, 1,
                                   alpha=0.5))
ax.set_xlim(-4, 8)
ax.set_ylim(-0.2, 8)

x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 8])

from scipy.stats import norm
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 5])





"""
Decision Tree and Random Forest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

data = pd.DataFrame({"toothed":["True","True","True","False","True","True","True","True","True","False"],
                     "hair":["True","True","False","True","True","True","False","False","True","False"],
                     "breathes":["True","True","True","True","True","True","False","True","True","True"],
                     "legs":["True","True","False","True","True","True","False","False","True","True"],
                     "species":["Mammal","Mammal","Reptile","Mammal","Mammal","Mammal","Reptile","Reptile","Mammal","Reptile"]}, 
                    columns=["toothed","hair","breathes","legs","species"])
features = data[["toothed","hair","breathes","legs"]]
target = data["species"]
data.describe()

#### Gini Index Split
p_mammal = data.loc[data.species=="Mammal"].shape[0]/data.shape[0]

## Use Toothed
toothed_true = data.loc[data.toothed=="True"]
p_mammal_toothed_true = toothed_true.loc[data.species=="Mammal"].shape[0]/toothed_true.shape[0]
gini_sub_true = p_mammal_toothed_true**2 + (1-p_mammal_toothed_true)**2

toothed_false = data.loc[data.toothed=="False"]
p_mammal_toothed_false = toothed_false.loc[data.species=="Mammal"].shape[0]/toothed_false.shape[0]
gini_sub_false = p_mammal_toothed_false**2 + (1-p_mammal_toothed_false)**2

gini_toothed = (toothed_true.shape[0]*gini_sub_true+toothed_false.shape[0]*gini_sub_false)/data.shape[0]
gini_toothed ## 0.525

## Use Hair
hair_true = data.loc[data.hair=="True"]
p_mammal_hair_true = hair_true.loc[data.species=="Mammal"].shape[0]/hair_true.shape[0]
gini_sub_true = p_mammal_hair_true**2 + (1-p_mammal_hair_true)**2

hair_false = data.loc[data.hair=="False"]
p_mammal_hair_false = hair_false.loc[data.species=="Mammal"].shape[0]/hair_false.shape[0]
gini_sub_false = p_mammal_hair_false**2 + (1-p_mammal_hair_false)**2

gini_hair = (hair_true.shape[0]*gini_sub_true+hair_false.shape[0]*gini_sub_false)/data.shape[0]
gini_hair ## 1

## Use Breathes
breathes_true = data.loc[data.breathes=="True"]
p_mammal_breathes_true = breathes_true.loc[data.species=="Mammal"].shape[0]/breathes_true.shape[0]
gini_sub_true = p_mammal_breathes_true**2 + (1-p_mammal_breathes_true)**2

breathes_false = data.loc[data.breathes=="False"]
p_mammal_breathes_false = breathes_false.loc[data.species=="Mammal"].shape[0]/breathes_false.shape[0]
gini_sub_false = p_mammal_breathes_false**2 + (1-p_mammal_breathes_false)**2

gini_breathes = (breathes_true.shape[0]*gini_sub_true+breathes_false.shape[0]*gini_sub_false)/data.shape[0]
gini_breathes ## 0.6

## Use Legs
legs_true = data.loc[data.legs=="True"]
p_mammal_legs_true = legs_true.loc[data.species=="Mammal"].shape[0]/legs_true.shape[0]
gini_sub_true = p_mammal_legs_true**2 + (1-p_mammal_legs_true)**2

legs_false = data.loc[data.legs=="False"]
p_mammal_legs_false = legs_false.loc[data.species=="Mammal"].shape[0]/legs_false.shape[0]
gini_sub_false = p_mammal_legs_false**2 + (1-p_mammal_legs_false)**2

gini_legs = (legs_true.shape[0]*gini_sub_true+legs_false.shape[0]*gini_sub_false)/data.shape[0]
gini_legs ## 0.829
## The highest Gini Index wins to be selected as split, so Legs are used.
### CART use Gini Index to create binary split
#### End Gini Index Split


#### Information Gain Split
## Entropy
p_mammal = data.loc[data.species=="Mammal"].shape[0]/data.shape[0]
p_reptile = 1 - p_mammal
h_mammal = p_mammal*np.log2(p_mammal)
h_reptile = p_reptile*np.log2(p_reptile)
H = -(h_mammal + h_reptile) 

## Toothed 
data_toothed = data.loc[data.toothed=="True"] ## Split the data by Toothed and calculate entropy for each split
p_mammal = data_toothed.loc[data_toothed.species=="Mammal"].shape[0]/data_toothed.shape[0]
p_reptile = 1 - p_mammal
h_mammal = p_mammal*np.log2(p_mammal)
h_reptile = p_reptile*np.log2(p_reptile)
H_toothed_true = -(h_mammal + h_reptile)

data_toothed = data.loc[data.toothed=="False"]
p_mammal = data_toothed.loc[data_toothed.species=="Mammal"].shape[0]/data_toothed.shape[0]
p_reptile = 1 - p_mammal
h_mammal = p_mammal*np.log2(p_mammal)
h_reptile = p_reptile*np.log2(p_reptile)
H_toothed_false = -(h_mammal + h_reptile)

p_toothed = data.loc[data.toothed=="True"].shape[0]/data.shape[0]
H_toothed_final = p_toothed*H_toothed_true + (1 - p_toothed)*H_toothed_false ## Using the ratio of how the data is split to calculate final entropy
info_gain_toothed = H - H_toothed_final
info_gain_toothed ## 0.0074

## breathes
data_breathes = data.loc[data.breathes=="True"]
p_mammal = data_breathes.loc[data_breathes.species=="Mammal"].shape[0]/data_breathes.shape[0]
p_reptile = 1 - p_mammal
h_mammal = p_mammal*np.log2(p_mammal)
h_reptile = p_reptile*np.log2(p_reptile)
H_breathes_true = -(h_mammal + h_reptile)

data_breathes = data.loc[data.breathes=="False"]
p_mammal = data_breathes.loc[data_breathes.species=="Mammal"].shape[0]/data_breathes.shape[0]
p_reptile = 1 - p_mammal
h_mammal = p_mammal*np.log2(p_mammal)
h_reptile = p_reptile*np.log2(p_reptile)
H_breathes_false = -(h_mammal + h_reptile)

p_breathes = data.loc[data.breathes=="True"].shape[0]/data.shape[0]
H_breathes_final = p_breathes*H_breathes_true
info_gain_breathes = H - H_breathes_final
info_gain_breathes ## 0.1445

## legs
data_legs = data.loc[data.legs=="True"]
p_mammal = data_legs.loc[data_legs.species=="Mammal"].shape[0]/data_legs.shape[0]
p_reptile = 1 - p_mammal
h_mammal = p_mammal*np.log2(p_mammal)
h_reptile = p_reptile*np.log2(p_reptile)
H_legs_true = -(h_mammal + h_reptile)

data_legs = data.loc[data.legs=="False"]
p_mammal = data_legs.loc[data_legs.species=="Mammal"].shape[0]/data_legs.shape[0]
p_reptile = 1 - p_mammal
h_mammal = p_mammal*np.log2(p_mammal)
h_reptile = p_reptile*np.log2(p_reptile)
H_legs_false = -(h_mammal + h_reptile)

p_legs = data.loc[data.legs=="True"].shape[0]/data.shape[0]
H_legs_final = p_legs*H_legs_true
info_gain_legs = H - H_legs_final
info_gain_legs ## 0.5568
## The largest information gain wins to be selected to split, so Legs are used to split

#### End Information Gain Split

#### Chi-Square Split
p_mammal = data.loc[data.species=="Mammal"].shape[0]/data.shape[0]

## Use Toothed
data_toothed = data.loc[data.toothed=="True"]
num_toothed_true = data_toothed.loc[data_toothed.species=="toothed"].shape[0]
num_toothed_true_exp = data_toothed.shape[0]*p_toothed
num_non_toothed_true = data_toothed.loc[data_toothed.species!="toothed"].shape[0]
num_non_toothed_true_exp = data_toothed.shape[0]*(1-p_toothed)
chi_toothed_true = np.sqrt((num_toothed_true-num_toothed_true_exp)**2/num_toothed_true_exp)
chi_non_toothed_true = np.sqrt((num_non_toothed_true-num_non_toothed_true_exp)**2/num_non_toothed_true_exp)

data_toothed = data.loc[data.toothed=="False"]
num_toothed_false = data_toothed.loc[data_toothed.species=="toothed"].shape[0]
num_toothed_false_exp = data_toothed.shape[0]*p_toothed
num_non_toothed_false = data_toothed.loc[data_toothed.species!="toothed"].shape[0]
num_non_toothed_false_exp = data_toothed.shape[0]*(1-p_toothed)
chi_toothed_false = np.sqrt((num_toothed_false-num_toothed_false_exp)**2/num_toothed_false_exp)
chi_non_toothed_false = np.sqrt((num_toothed_false-num_toothed_false_exp)**2/num_toothed_false_exp)

chi_toothed = chi_toothed_true+chi_non_toothed_true+chi_toothed_false+chi_non_toothed_false ## 0.568

## Use Breathes
data_breathes = data.loc[data.breathes=="True"]
num_breathes_true = data_breathes.loc[data_breathes.species=="breathes"].shape[0]
num_breathes_true_exp = data_breathes.shape[0]*p_breathes
num_non_breathes_true = data_breathes.loc[data_breathes.species!="breathes"].shape[0]
num_non_breathes_true_exp = data_breathes.shape[0]*(1-p_breathes)
chi_breathes_true = np.sqrt((num_breathes_true-num_breathes_true_exp)**2/num_breathes_true_exp)
chi_non_breathes_true = np.sqrt((num_non_breathes_true-num_non_breathes_true_exp)**2/num_non_breathes_true_exp)

data_breathes = data.loc[data.breathes=="False"]
num_breathes_false = data_breathes.loc[data_breathes.species=="breathes"].shape[0]
num_breathes_false_exp = data_breathes.shape[0]*p_breathes
num_non_breathes_false = data_breathes.loc[data_breathes.species!="breathes"].shape[0]
num_non_breathes_false_exp = data_breathes.shape[0]*(1-p_breathes)
chi_breathes_false = np.sqrt((num_breathes_false-num_breathes_false_exp)**2/num_breathes_false_exp)
chi_non_breathes_false = np.sqrt((num_breathes_false-num_breathes_false_exp)**2/num_breathes_false_exp)

chi_breathes = chi_breathes_true+chi_non_breathes_true+chi_breathes_false+chi_non_breathes_false ## 13.282

## Use Legs
data_legs = data.loc[data.legs=="True"]
num_legs_true = data_legs.loc[data_legs.species=="legs"].shape[0]
num_legs_true_exp = data_legs.shape[0]*p_legs
num_non_legs_true = data_legs.loc[data_legs.species!="legs"].shape[0]
num_non_legs_true_exp = data_legs.shape[0]*(1-p_legs)
chi_legs_true = np.sqrt((num_legs_true-num_legs_true_exp)**2/num_legs_true_exp)
chi_non_legs_true = np.sqrt((num_non_legs_true-num_non_legs_true_exp)**2/num_non_legs_true_exp)

data_legs = data.loc[data.legs=="False"]
num_legs_false = data_legs.loc[data_legs.species=="legs"].shape[0]
num_legs_false_exp = data_legs.shape[0]*p_legs
num_non_legs_false = data_legs.loc[data_legs.species!="legs"].shape[0]
num_non_legs_false_exp = data_legs.shape[0]*(1-p_legs)
chi_legs_false = np.sqrt((num_legs_false-num_legs_false_exp)**2/num_legs_false_exp)
chi_non_legs_false = np.sqrt((num_legs_false-num_legs_false_exp)**2/num_legs_false_exp)

chi_legs = chi_legs_true+chi_non_legs_true+chi_legs_false+chi_non_legs_false ## 8.493

## Chi-Square from Breathes split has the highest to win
#### End Chi-Square Split


iris = load_iris()
X = iris.data
y = iris.target

digits = load_digits()
X = digits.data
y = digits.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=0, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier
tree_gini = DecisionTreeClassifier(criterion='gini',max_depth=3)
tree_gini.fit(Xtrain, ytrain)
y_pred_gini = tree_gini.predict(Xtest)
accuracy_score(ytest,y_pred_gini)
mat_gini = confusion_matrix(ytest, y_pred_gini)
classification_report(ytest, y_pred_gini)

tree_entropy = DecisionTreeClassifier(criterion='entropy',max_depth=3)
tree_entropy.fit(Xtrain, ytrain)
y_pred_entropy = tree_entropy.predict(Xtest)
accuracy_score(ytest,y_pred_entropy)
mat_entropy = confusion_matrix(ytest, y_pred_entropy)
classification_report(ytest, y_pred_entropy)

## Using GridSearchCV for optimizing parameters
from sklearn.grid_search import GridSearchCV
sample_split_range = list(range(2, 51))
param_grid = dict(min_samples_split=sample_split_range)
grid = GridSearchCV(tree_gini, param_grid, cv=10, scoring='accuracy')
grid.fit(Xtrain, ytrain)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

param_dist = {'max_depth': [2, 3, 4],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'min_samples_split': [2, 3, 5],
              'criterion': ['gini', 'entropy']}
dt = DecisionTreeClassifier()
cv_dt = GridSearchCV(dt, cv = 10, param_grid=param_dist, n_jobs = 3)
cv_dt.fit(Xtrain, ytrain)
cv_dt.best_params_
dt.set_params(criterion = 'entropy', max_depth = 4, max_features = None, min_samples_split = 3)
dt.fit(Xtrain, ytrain)

from sklearn.ensemble import RandomForestClassifier
rf_gini = RandomForestClassifier(criterion = 'gini', n_estimators = 1000, random_state = 1)
rf_gini.fit(Xtrain, ytrain)
y_pred_gini = rf_gini.predict(Xtest)
accuracy_score(ytest,y_pred_gini)
rf.feature_importances_




"""
Logistics Regression
"""
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

iris = load_iris()
type(iris)
print(iris.data)
print(type(iris.data))
print(type(iris.target))
print(iris.data.shape)
X = iris.data
y = iris.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=0, test_size = 0.3)

lr = LogisticRegression(C=100)
lr.fit(Xtrain, ytrain)
lr.coef_
lr.intercept_
np.exp(lr.coef_)
y_pred = lr.predict(Xtest)
y_prob = lr.predict_proba(Xtest)
accuracy_score(ytest,y_pred)
confusion_matrix(ytest,y_pred)
classification_report(ytest,y_pred)
fpr, tpr, thresholds = roc_curve(ytest, y_pred, pos_label=2)
roc_auc_score(ytest, y_pred)

y_pred_cv = cross_validation.cross_val_predict(lr, X, y, cv=5)
accuracy_score(y, y_pred_cv)




"""
Naive Bayes 
"""

# the tuples consist of (delay time of train1, number of times)
# tuples are (minutes, number of times)
in_time = [(0, 22), (1, 19), (2, 17), (3, 18),
           (4, 16), (5, 15), (6, 9), (7, 7),
           (8, 4), (9, 3), (10, 3), (11, 2)]
too_late = [(6, 6), (7, 9), (8, 12), (9, 17), 
            (10, 18), (11, 15), (12,16), (13, 7),
            (14, 8), (15, 5)]

%matplotlib inline
import matplotlib.pyplot as plt
X, Y = zip(*in_time)
X2, Y2 = zip(*too_late)
bar_width = 0.9
plt.bar(X, Y, bar_width,  color="blue", alpha=0.75, label="in time")
bar_width = 0.8
plt.bar(X2, Y2, bar_width,  color="red", alpha=0.75, label="too late")
plt.legend(loc='upper right')
plt.show()

in_time_dict = dict(in_time)
too_late_dict = dict(too_late)
def catch_the_train(min):
    s = in_time_dict.get(min, 0)
    if s == 0:
        return 0
    else:
        m = too_late_dict.get(min, 0)
        return s / (s + m)
for minutes in range(-1, 13):
    print(minutes, catch_the_train(minutes))
    
import numpy as np
genders = ["male", "female"]
persons = []
with open("C:/Users/f371528/Desktop/QC/Python/person_data.txt") as fh:
    for line in fh:
        persons.append(line.strip().split())
firstnames = {}
heights = {}
for gender in genders:
    firstnames[gender] = [ x[0] for x in persons if x[4]==gender]
    heights[gender] = [ x[2] for x in persons if x[4]==gender]
    heights[gender] = np.array(heights[gender], np.int)
    
for gender in ("female", "male"):
    print(gender + ":")
    print(firstnames[gender][:10])
    print(heights[gender][:10])




"""
Statistical Tests
"""
from scipy import stats

###### Normality Tests
## Shapiro-Wilk Test - Tests whether a data sample has a Gaussian distribution
data = stats.norm.rvs(loc=5, scale=3, size=100)
data = [1,2,3,4,5,6,7,8,9,10]
stat, p = stats.shapiro(data)

## D’Agostino’s K^2 Test - Tests whether a data sample has a Gaussian distribution
from scipy.stats import normaltest
data = [1,2,3,4,5,6,7,8,9,10]
stat, p = normaltest(data)

## Anderson-Darling Test - Tests whether a data sample has a Gaussian distribution
result = stats.anderson(data)

###### End Normality Tests

###### Correlation Tests
## Pearson’s Correlation Coefficient - Tests whether two samples have a linear relationship
data1 = [1,2,3,4,5]
data2 = [6,7,8,9,10]
corr, p = stats.pearsonr(data1, data2)

## Spearman’s Rank Correlation - Tests whether two samples have a monotonic relationship
corr, p = stats.spearmanr(data1, data2)

## Kendall’s Rank Correlation - Tests whether two samples have a monotonic relationship
corr, p = stats.kendalltau(data1, data2)

## Chi-Squared Test - Tests whether two categorical variables are related or independent
stat, p, dof, expected = stats.chi2_contingency(table)

###### End Correlation Tests

###### Parametric Statistical Hypothesis Tests
## Student’s t-test - Tests whether the means of two independent samples are significantly different
stat, p = stats.ttest_ind(data1, data2)

## Paired Student’s t-test - Tests whether the means of two paired samples are significantly different
stat, p = stats.ttest_rel(data1, data2)

## Analysis of Variance Test (ANOVA) - Tests whether the means of two or more independent samples are significantly different
stat, p = stats.f_oneway(data1, data2, ...)

## Repeated Measures ANOVA Test - Tests whether the means of two or more paired samples are significantly different

###### End Parametric Statistical Hypothesis Tests

###### Nonparametric Statistical Hypothesis Tests
## Mann-Whitney U Test - Tests whether the distributions of two independent samples are equal or not
stat, p = stats.mannwhitneyu(data1, data2)

## Wilcoxon Signed-Rank Test - Tests whether the distributions of two paired samples are equal or not
stat, p = stats.wilcoxon(data1, data2)

## Kruskal-Wallis H Test - Tests whether the distributions of two or more independent samples are equal or not
stat, p = stats.kruskal(data1, data2, ...)

## Friedman Test - Tests whether the distributions of two or more paired samples are equal or not
stat, p = stats.friedmanchisquare(data1, data2, ...)

###### End Nonparametric Statistical Hypothesis Tests





"""
A/B Testing
"""
import numpy as np
import math as mt
from scipy.stats import norm
import pandas as pd

#Let's place this estimators into a dictionary for ease of use later
baseline = {"Cookies":40000,"Clicks":3200,"Enrollments":660,"CTP":0.08,"GConversion":0.20625,
           "Retention":0.53,"NConversion":0.109313}

#Scale The counts estimates
baseline["Cookies"] = 5000
baseline["Clicks"]=baseline["Clicks"]*(5000/40000)
baseline["Enrollments"]=baseline["Enrollments"]*(5000/40000)
baseline

# Let's get the p and n we need for Gross Conversion (GC)
# and compute the Stansard Deviation(sd) rounded to 4 decimal digits.
GC={}
GC["d_min"]=0.01
GC["p"]=baseline["GConversion"]
#p is given in this case - or we could calculate it from enrollments/clicks
GC["n"]=baseline["Clicks"]
GC["sd"]=round(np.sqrt((GC["p"]*(1-GC["p"]))/GC["n"]),4)
GC["sd"]

# Let's get the p and n we need for Retention(R)
# and compute the Stansard Deviation(sd) rounded to 4 decimal digits.
R={}
R["d_min"]=0.01
R["p"]=baseline["Retention"]
R["n"]=baseline["Enrollments"]
R["sd"]=round(np.sqrt((R["p"]*(1-R["p"]))/R["n"]),4)
R["sd"]

# Let's get the p and n we need for Net Conversion (NC)
# and compute the Standard Deviation (sd) rounded to 4 decimal digits.
NC={}
NC["d_min"]=0.0075
NC["p"]=baseline["NConversion"]
NC["n"]=baseline["Clicks"]
NC["sd"]=round(np.sqrt((NC["p"]*(1-NC["p"]))/NC["n"]),4)
NC["sd"]

def get_sds(p,d):
    sd1=np.sqrt(2*p*(1-p))
    sd2=np.sqrt(p*(1-p)+(p+d)*(1-(p+d)))
    x=[sd1,sd2]
    return x

#Inputs: required alpha value (alpha should already fit the required test)
#Returns: z-score for given alpha
def get_z_score(alpha):
    return norm.ppf(alpha)

# Inputs p-baseline conversion rate which is our estimated p and d-minimum detectable change
# Returns
def get_sds(p,d):
    sd1=np.sqrt(2*p*(1-p))
    sd2=np.sqrt(p*(1-p)+(p+d)*(1-(p+d)))
    sds=[sd1,sd2]
    return sds

# Inputs:sd1-sd for the baseline,sd2-sd for the expected change,alpha,beta,d-d_min,p-baseline estimate p
# Returns: the minimum sample size required per group according to metric denominator
def get_sampSize(sds,alpha,beta,d):
    n=pow((get_z_score(1-alpha/2)*sds[0]+get_z_score(1-beta)*sds[1]),2)/pow(d,2)
    return n

GC["d"]=0.01
R["d"]=0.01
NC["d"]=0.0075

GC["SampSize"]=round(get_sampSize(get_sds(GC["p"],GC["d"]),0.05,0.2,GC["d"]))
GC["SampSize"]
GC["SampSize"]=round(GC["SampSize"]/0.08*2)
GC["SampSize"]

# Getting a nice integer value
R["SampSize"]=round(get_sampSize(get_sds(R["p"],R["d"]),0.05,0.2,R["d"]))
R["SampSize"]
R["SampSize"]=R["SampSize"]/0.08/0.20625*2
R["SampSize"]

# Getting a nice integer value
NC["SampSize"]=round(get_sampSize(get_sds(NC["p"],NC["d"]),0.05,0.2,NC["d"]))
NC["SampSize"]
NC["SampSize"]=NC["SampSize"]/0.08*2
NC["SampSize"]

# we use pandas to load datasets
control=pd.read_csv("C:/Users/f371528/Desktop/QC/Python/control_data.csv")
experiment=pd.read_csv("C:/Users/f371528/Desktop/QC/Python/experiment_data.csv")
control.head()

pageviews_cont=control['Pageviews'].sum()
pageviews_exp=experiment['Pageviews'].sum()
pageviews_total=pageviews_cont+pageviews_exp
print ("number of pageviews in control:", pageviews_cont)
print ("number of Pageviewsin experiment:" ,pageviews_exp)

p=0.5
alpha=0.05
p_hat=round(pageviews_cont/(pageviews_total),4)
sd=np.sqrt(p*(1-p)/(pageviews_total))
ME=round(get_z_score(1-(alpha/2))*sd,4)
print ("The confidence interval is between",p-ME,"and",p+ME,"; Is",p_hat,"inside this range?")

clicks_cont=control['Clicks'].sum()
clicks_exp=experiment['Clicks'].sum()
clicks_total=clicks_cont+clicks_exp
ctp_cont=clicks_cont/pageviews_cont
ctp_exp=clicks_exp/pageviews_exp
d_hat=round(ctp_exp-ctp_cont,4)
p_pooled=clicks_total/pageviews_total
sd_pooled=np.sqrt(p_pooled*(1-p_pooled)*(1/pageviews_cont+1/pageviews_exp))
ME=round(get_z_score(1-(alpha/2))*sd_pooled,4)
print ("The confidence interval is between",0-ME,"and",0+ME,"; Is",d_hat,"within this range?")

# Count the total clicks from complete records only
clicks_cont=control["Clicks"].loc[control["Enrollments"].notnull()].sum()
clicks_exp=experiment["Clicks"].loc[experiment["Enrollments"].notnull()].sum()

#Gross Conversion - number of enrollments divided by number of clicks
enrollments_cont=control["Enrollments"].sum()
enrollments_exp=experiment["Enrollments"].sum()

GC_cont=enrollments_cont/clicks_cont
GC_exp=enrollments_exp/clicks_exp
GC_pooled=(enrollments_cont+enrollments_exp)/(clicks_cont+clicks_exp)
GC_sd_pooled=np.sqrt(GC_pooled*(1-GC_pooled)*(1/clicks_cont+1/clicks_exp))
GC_ME=round(get_z_score(1-alpha/2)*GC_sd_pooled,4)
GC_diff=round(GC_exp-GC_cont,4)
print("The change due to the experiment is",GC_diff*100,"%")
print("Confidence Interval: [",GC_diff-GC_ME,",",GC_diff+GC_ME,"]")
print ("The change is statistically significant if the CI doesn't include 0. In that case, it is practically significant if",-GC["d_min"],"is not in the CI as well.")

#Net Conversion - number of payments divided by number of clicks
payments_cont=control["Payments"].sum()
payments_exp=experiment["Payments"].sum()

NC_cont=payments_cont/clicks_cont
NC_exp=payments_exp/clicks_exp
NC_pooled=(payments_cont+payments_exp)/(clicks_cont+clicks_exp)
NC_sd_pooled=np.sqrt(NC_pooled*(1-NC_pooled)*(1/clicks_cont+1/clicks_exp))
NC_ME=round(get_z_score(1-alpha/2)*NC_sd_pooled,4)
NC_diff=round(NC_exp-NC_cont,4)
print("The change due to the experiment is",NC_diff*100,"%")
print("Confidence Interval: [",NC_diff-NC_ME,",",NC_diff+NC_ME,"]")
print ("The change is statistically significant if the CI doesn't include 0. In that case, it is practically significant if",NC["d_min"],"is not in the CI as well.")

# Sign Test
#let's first create the dataset we need for this:
# start by merging the two datasets
full=control.join(other=experiment,how="inner",lsuffix="_cont",rsuffix="_exp")
#Let's look at what we got
full.count()
#now we only need the complete data records
full=full.loc[full["Enrollments_cont"].notnull()]
full.count()

# Perfect! Now, derive a new column for each metric, so we have it's daily values
# We need a 1 if the experiment value is greater than the control value=
x=full['Enrollments_cont']/full['Clicks_cont']
y=full['Enrollments_exp']/full['Clicks_exp']
full['GC'] = np.where(x<y,1,0)
# The same now for net conversion
z=full['Payments_cont']/full['Clicks_cont']
w=full['Payments_exp']/full['Clicks_exp']
full['NC'] = np.where(z<w,1,0)
full.head()

GC_x=full.GC[full["GC"]==1].count()
NC_x=full.NC[full["NC"]==1].count()
n=full.NC.count()
print("No. of cases for GC:",GC_x,'\n',
      "No. of cases for NC:",NC_x,'\n',
      "No. of total cases",n)

#first a function for calculating probability of x=number of successes
def get_prob(x,n):
    p=round(mt.factorial(n)/(mt.factorial(x)*mt.factorial(n-x))*0.5**x*0.5**(n-x),4)
    return p
#next a function to compute the pvalue from probabilities of maximum x
def get_2side_pvalue(x,n):
    p=0
    for i in range(0,x+1):
        p=p+get_prob(i,n)
    return 2*p

print ("GC Change is significant if",get_2side_pvalue(GC_x,n),"is smaller than 0.05")
print ("NC Change is significant if",get_2side_pvalue(NC_x,n),"is smaller than 0.05")








from pymc import Uniform, rbernoulli, Bernoulli, MCMC
from matplotlib import pyplot as plt
import numpy as np

# true value of p_A (unknown)
p_A_true = 0.05
# number of users visiting page A
N = 1500
occurrences = rbernoulli(p_A_true, N)

print('Click-BUY:')
print(occurrences.sum())
print('Observed frequency:')
print(occurrences.sum() / float(N))


"""
Introduction to NumPy
"""

import numpy as np
np.__version__

import array
L = list(range(10))
A = array.array('i', L)
A

# integer array:
np.array([1, 4, 2, 5, 3])

np.array([3.14, 4, 2, 3])
list(np.array([3.14,1,2,3,4]))

np.array([1, 2, 3, 4], dtype='float32')

np.array([range(i, i + 3) for i in [2, 4, 6]])

# Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)
np.zeros([10,10])
np.zeros((10,10))

# Create a 3x5 floating-point array filled with ones
np.ones((3, 5), dtype=float)

# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)

# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0, 20, 2)

# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3, 3))

# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))

# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))

# Create a 3x3 identity matrix
np.eye(3)

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)

"""
Data Types
"""

np.zeros(10, dtype='int16')
np.zeros(10, dtype=np.int16)

# seed for reproducibility
np.random.seed(0)

x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))

x1.ndim
x2.shape
x3.size
x3.dtype
x3.itemsize
x3.nbytes
print("nbytes = size * itemsize:", x3.nbytes, "=", x3.size, "*", x3.itemsize)

x1[-1]
x2[0,0]
x3[0,0,0]

np.arange(1, 10).reshape(3,3)

##Concatenation of arrays
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = [99, 99, 99]
np.concatenate([x, y])
np.concatenate([x, y, z])
print(np.concatenate([x, y, z]))

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
np.concatenate([grid, grid])
np.vstack([grid, grid])
np.hstack([grid, grid])
np.concatenate([grid, grid], axis=1)
np.concatenate([grid, grid], axis=0)

np.dstack([grid, grid])

##Splitting of arrays
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])

y = np.reshape(x, (4,2))
y1, y2 = np.hsplit(y, [1])
y1, y2 = np.vsplit(y, [2])

np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
        
values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)

big_array = np.random.randint(1, 100, size=1000000)
%timeit compute_reciprocals(big_array)

#### Binning
np.random.seed(42)
x = np.random.randn(100)

# compute a histogram by hand
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

# find the appropriate bin for each x
i = np.searchsorted(bins, x)

# add 1 to each of these bins
np.add.at(counts, i, 1)

#### Sorting
x = np.array([2, 1, 4, 3, 5])
np.sort(x)

x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)

### Partitioning
x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)


#### Structured Array
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

x = np.zeros(4, dtype=int)
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),'formats':('U10', 'i4', 'f8')})
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

# Get all names
data['name']

# Get first row of data
data[0]

# Get the name from the last row
data[-1]['name']

# Get names where age is under 30
data[data['age'] < 30]['name']




"""
Introduction to Pandas
"""
import numpy as np
import pandas as pd
pd.__version__

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
data.values
data.index
data[1:3]
data.values[1:3]

data = pd.Series([0.25, 0.5, 0.75, 1.0],index=['a', 'b', 'c', 'd'])
data
data['b']

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population
population['California']
population['California':'Illinois']

pd.Series(5, index=[100, 200, 300])
pd.Series({2:'a', 1:'b', 3:'c'})
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area

states = pd.DataFrame({'population': population,
                       'area': area})
states
states.values
states.index
states.columns
states['area']
states['col0']

pd.DataFrame(population, columns=['population'])

data = [{'a': i, 'b': 2 * i} for i in range(3)]
pd.DataFrame(data)
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])

ind = pd.Index([2, 3, 5, 7, 11])
ind
ind[::2]
print(ind.size, ind.shape, ind.ndim, ind.dtype)

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB  # intersection
indA | indB  # union
indA ^ indB  # symmetric difference

## Missing Values
vals1 = np.array([1, None, 3, 4])
vals1.dtype

vals2 = np.array([1, np.nan, 3, 4]) 
vals2.dtype

1 + np.nan

data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data.notnull()
data.dropna()

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df
df.isnull()
df.notnull()
df.dropna()
df.dropna(axis='columns')
df.dropna(axis=0)
df.dropna(axis=1)
df[3] = np.nan
df
df.dropna(axis='columns', how='all')
df.dropna(axis='rows', thresh=2)

data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)
data.fillna(method='ffill')
data.fillna(method='bfill')
df.fillna(method='ffill', axis=1)

## Indexing
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
pop
index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)
pop
pop[:, 2010]
pop_df = pop.unstack()
pop_df
pop_1 = pop_df.stack()
pop_1
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
pop_df
f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()

df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
df
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
pop.index.names = ['state', 'year']
pop

# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
index
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])
columns
# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
health_data['Guido']
health_data = health_data.sort_index()
health_data

## Combining Datasets: Concat and Append, Merge and Join
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])

df1 = pd.make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
display('df1', 'df2', 'pd.concat([df1, df2])')

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.merge(df1, df2, on='employee')
df3
df4 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
pd.merge(df1, df4, left_on="employee", right_on="name").drop('name', axis=1)

df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
pd.merge(df1a, df2a, left_index=True, right_index=True)
df1a.join(df2a)
pd.merge(df1a, df4, left_index=True, right_on='name')

df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])
pd.merge(df6, df7, how='inner')
pd.merge(df6, df7, how='outer')

##Aggregation
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
ser
ser.sum()
ser.mean()

df = pd.DataFrame({'A': rng.rand(5),
                   'B': rng.rand(5)})
df
df.mean()
df.mean(axis='columns')

df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])
df
df.groupby('key')
df.groupby('key').sum()

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['key', 'data1', 'data2'])
df
df.groupby('key').aggregate(['min', np.median, max])
df.groupby('key').aggregate({'data1': 'min','data2': 'max'})
df.groupby('key').transform(lambda x: x - x.mean())

##Vectorized String Operations
x = np.array([2, 3, 5, 7, 11, 13])
x * 2

data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]
names = pd.Series(data)
names
names.str.capitalize()


####Working with Time Series
from datetime import datetime
datetime(year=2015, month=7, day=4)

date = np.array('2015-07-04', dtype=np.datetime64)
date
date + np.arange(12)

#Day-based datetime
np.datetime64('2015-07-04')
#Minute-based datetime
np.datetime64('2015-07-04 12:00')
##Nanosecond-based datetime
np.datetime64('2015-07-04 12:59:59.50', 'ns')

date = pd.to_datetime("4th of July, 2015")
date
date.strftime('%A')
date + pd.to_timedelta(np.arange(12), 'D')

index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)
data
data['2014-07-04':'2015-07-04']
data['2015']

dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                       '2015-Jul-6', '07-07-2015', '20150708'])
dates
dates.to_period('D')
dates - dates[0]
pd.date_range('2015-07-03', '2015-07-10')
pd.date_range('2015-07-03', periods=8)
pd.date_range('2015-07-03', periods=8, freq='H')
pd.period_range('2015-07', periods=8, freq='M')
pd.timedelta_range(0, periods=10, freq='H')
pd.timedelta_range(0, periods=9, freq="2H30T")

from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=5, freq=BDay())

conda install pandas-datareader
from pandas_datareader import data
goog = data.DataReader('GOOG', start='2004', end='2016',data_source='google')
goog.head()




"""
Check the Up & Downs of US Stock Market

"""

import pandas, datetime, matplotlib

#df = pandas.read_csv("C:\\Users\\f371528\\Desktop\\DJCA.csv")
#df = pandas.read_csv("C:\\Users\\f371528\\Desktop\\DJCA.csv", na_values=['.'])
def cal_weekday(row):
  if row['Weekday'] == 1:
      return 'Monday'
  elif row['Weekday'] == 2:
      return 'Tuesday'
  elif row['Weekday'] == 3:
      return 'Wednesday'
  elif row['Weekday'] == 4:
      return 'Thursday'
  else:
      return 'Friday'

def direction(row):
    if row['Diff'] >= 0:
        return 'Up'
    elif row['Diff'] < 0:
        return 'Down'
    else:
        return 'Flat'

url_dow = "C:\\Users\\f371528\\Desktop\\DJCA.csv"
url_nas = "C:\\Users\\f371528\\Desktop\\NASDAQCOM.csv"
url_sp = "C:\\Users\\f371528\\Desktop\\SP500.csv"
headers = ['Date', 'Value']
dtypes = {'Date': 'str', 'Value': 'float'}


st_type = ['dow', 'nas', 'sp']
for i in st_type:
    if i == 'dow':
        df_dow = pandas.read_table(url_dow, sep = ',', header = 0, names = headers, dtype = dtypes, na_values=["."], parse_dates=['Date'])
        df_dow['Weekday'] = df_dow.apply(lambda row: datetime.date.weekday(row['Date']), axis = 1) + 1
        df_dow['Year'] = df_dow.apply(lambda row: (row['Date']).year, axis = 1)
        df1_dow = df_dow.dropna()
        df1_dow = df1_dow.sort_values(by = ['Date'])
        df1_dow['Diff'] = df1_dow.diff()['Value']
        df1_dow = df1_dow[df1_dow.Year >= 2009]
        df1_dow['weekday'] = df1_dow.apply(lambda row: cal_weekday(row), axis = 1)
        df1_dow['Direction'] = df1_dow.apply(lambda row: direction(row), axis = 1)
        df1_dow = df1_dow.dropna()
    elif i == 'nas':
        df_nas = pandas.read_table(url_nas, sep = ',', header = 0, names = headers, dtype = dtypes, na_values=["."], parse_dates=['Date'])
        df_nas['Weekday'] = df_nas.apply(lambda row: datetime.date.weekday(row['Date']), axis = 1) + 1
        df_nas['Year'] = df_nas.apply(lambda row: (row['Date']).year, axis = 1)
        df1_nas = df_nas.dropna()
        df1_nas = df1_nas.sort_values(by = ['Date'])
        df1_nas['Diff'] = df1_nas.diff()['Value']
        df1_nas = df1_nas[df1_nas.Year >= 2009]
        df1_nas['weekday'] = df1_nas.apply(lambda row: cal_weekday(row), axis = 1)
        df1_nas['Direction'] = df1_nas.apply(lambda row: direction(row), axis = 1)
        df1_nas = df1_nas.dropna()
    else:
        df_sp = pandas.read_table(url_sp, sep = ',', header = 0, names = headers, dtype = dtypes, na_values=["."], parse_dates=['Date'])
        df_sp['Weekday'] = df_sp.apply(lambda row: datetime.date.weekday(row['Date']), axis = 1) + 1
        df_sp['Year'] = df_sp.apply(lambda row: (row['Date']).year, axis = 1)
        df1_sp = df_sp.dropna()
        df1_sp = df1_sp.sort_values(by = ['Date'])
        df1_sp['Diff'] = df1_sp.diff()['Value']
        df1_sp = df1_sp[df1_sp.Year >= 2009]
        df1_sp['weekday'] = df1_sp.apply(lambda row: cal_weekday(row), axis = 1)
        df1_sp['Direction'] = df1_sp.apply(lambda row: direction(row), axis = 1)
        df1_sp = df1_sp.dropna()


ag_dow = df1_dow.groupby(['weekday','Direction'])['Date'].count().unstack().reset_index()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
mapping = {day: i for i, day in enumerate(weekdays)}
key = ag_dow['weekday'].map(mapping)    
ag1_dow = ag_dow.iloc[key.argsort()]
ag1_dow.plot(kind = 'bar', x = 'weekday', color = ['r', 'g']*5, width = 0.8, title = 'Dow Index from 2009 to 2018')

ag_nas = df1_nas.groupby(['weekday','Direction'])['Date'].count().unstack().reset_index()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
mapping = {day: i for i, day in enumerate(weekdays)}
key = ag_nas['weekday'].map(mapping)    
ag1_nas = ag_nas.iloc[key.argsort()]
ag1_nas.plot(kind = 'bar', x = 'weekday', color = ['r', 'g']*5, width = 0.8, title = 'NASDAQ Index from 2009 to 2018')

ag_sp = df1_sp.groupby(['weekday','Direction'])['Date'].count().unstack().reset_index()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
mapping = {day: i for i, day in enumerate(weekdays)}
key = ag_sp['weekday'].map(mapping)    
ag1_sp = ag_sp.iloc[key.argsort()]
ag1_sp.plot(kind = 'bar', x = 'weekday', color = ['r', 'g']*5, width = 0.8, title = 'SP500 Index from 2009 to 2018')

df.shape
df.ndim
df.head()
df.tail(3)
df.describe()
df.dtypes
print(df['Date'].dtypes)
print(df['Value'].dtypes)

df.sort_values(by='VALUE')
df.loc[0:6,['VALUE']]
df.iloc[3:5,:]
df.iat[1,1]




