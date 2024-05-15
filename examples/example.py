# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from ppi_py.cross_ppi import crossppi_ols_pred_ci
from sklearn.datasets import make_regression
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# %%
K = 5
x, y = make_regression(n_samples=5000, n_features=3, n_informative=3, coef=False)
l_x = x[:2000, :]
l_y = y[:2000]
u_x = x[2000:5000, :]
u_y = y[2000:5000]

# %%
l_y_hat = np.zeros(l_y.shape)
u_y_hat = np.zeros((u_y.shape[0], K))

kf = KFold(n_splits=K)
for i, (train_index, test_index) in enumerate(kf.split(l_x)):
    X_train, X_test = l_x[train_index], l_x[test_index]
    y_train, y_test = l_y[train_index], l_y[test_index]
    ml = RandomForestRegressor(random_state=i, max_features='sqrt', min_samples_leaf=3).fit(X_train, y_train)
    l_y_hat[test_index] += ml.predict(X_test)
    u_y_hat[:, i] = ml.predict(u_x)

# %%
res = crossppi_ols_pred_ci(l_x, l_y, l_y_hat, u_x, u_y_hat, u_x, alpha=0.05)
print("coverage rate (two-sided):", np.mean(np.logical_and(res[0] <= u_y, res[1] >= u_y)))
print("coverage rate (one-sided):", np.mean(res[0] <= u_y))
pass

