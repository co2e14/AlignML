from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle
import os

def train(trainingSet, modelType, degree=None):
    data = np.loadtxt(f"readings/{trainingSet}", delimiter=",")
    newdata = data.copy()
    newdata[1:, :3] -= data[0, :3]
    data = np.delete(newdata, 0, axis=0)
    
    X = data[:, :2]
    Y = data[:, 3:]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    if modelType == 'random_forest':
        model = RandomForestRegressor(n_estimators=100000, random_state=42)
    elif modelType == 'linear':
        model = LinearRegression()
    elif modelType == 'polynomial':
        if degree is None:
            raise ValueError("Degree must be specified for polynomial regression")
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif modelType == 'svr':
        model = MultiOutputRegressor(SVR())
    elif modelType == 'decision_tree':
        model = DecisionTreeRegressor()
    elif modelType == 'gradient_boosting':
        model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    elif modelType == 'adaboost':
        model = MultiOutputRegressor(AdaBoostRegressor(random_state=42))
    elif modelType == 'knn':
        model = MultiOutputRegressor(KNeighborsRegressor())
    elif modelType == 'ridge':
        model = Ridge()
    elif modelType == 'lasso':
        model = MultiOutputRegressor(Lasso())
    elif modelType == 'elasticnet':
        model = MultiOutputRegressor(ElasticNet())
    elif modelType == 'bayesian_ridge':
        model = MultiOutputRegressor(BayesianRidge())
    else:
        raise ValueError(f"Unknown model type: {modelType}")
    
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    mse = mean_absolute_error(Y_test, Y_pred)
    print(f"Model: {modelType}, MSE: {mse}")
    
    filename, _ = os.path.splitext(trainingSet)
    with open(f"./models/model_{modelType}_{filename}.pkl", "wb") as toDump:
        pickle.dump(model, toDump)

if __name__ == "__main__":
    if not os.path.exists("./models"):
        os.mkdir("./models")
    
    trainingSet = "rbvs_20062024.txt"
    
    models = [
        ('random_forest', None),
        ('linear', None),
        ('polynomial', 2),
        ('polynomial', 3),        
        ('polynomial', 4),
        ('svr', None),
        ('decision_tree', None),
        ('gradient_boosting', None),
        ('adaboost', None),
        ('knn', None),
        ('ridge', None),
        ('lasso', None),
        ('elasticnet', None),
        ('bayesian_ridge', None)
    ]
    
    for modelType, degree in models:
        train(trainingSet, modelType, degree)