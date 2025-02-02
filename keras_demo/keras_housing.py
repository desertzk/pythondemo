
# Regression Example With Boston Dataset: Standardized
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Regression Example With Boston Dataset: Standardized and Larger


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def base_test():
    # load dataset
    dataframe = read_csv("housing.data", delim_whitespace=True, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]
    # define base model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def larger_test():
    # 13 inputs -> [13 -> 6] -> 1 output
    # load dataset
    dataframe = read_csv("housing.data", delim_whitespace=True, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]
    # define the model
    def larger_model():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(6, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def wider_test():
    # load dataset
    dataframe = read_csv("housing.data", delim_whitespace=True, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]
    # define wider model
    def wider_model():
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))



base_test()
larger_test()
wider_test()