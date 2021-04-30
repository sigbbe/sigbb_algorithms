
from matplotlib import pyplot as plt


def gini(arr):
    # first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = len(arr)
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_


def lorenz_curve(X):
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    X_lorenz[0], X_lorenz[-1]
    fig, ax = plt.subplots(figsize=[6, 6])
    # scatter plot of Lorenz curve
    ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,
               marker='x', color='darkgreen', s=100)
    # line plot of equality
    ax.plot([0, 1], [0, 1], color='k')


print(gini(X_train))


%matplotlib inline

X = X_train
X_lorenz = X.cumsum() / X.sum()
X_lorenz = np.insert(X_lorenz, 0, 0)
X_lorenz[0], X_lorenz[-1]

fig, ax = plt.subplots(figsize=[6, 6])
# scatter plot of Lorenz curve
ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,
           marker='x', color='darkgreen', s=100)
# line plot of equality
ax.plot([0, 1], [0, 1], color='k')

X = np.append(np.random.poisson(lam=10, size=40),
              np.random.poisson(lam=100, size=10))

lorenz_curve(X)
