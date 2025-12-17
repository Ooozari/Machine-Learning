import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import random
import itertools
import optuna
import keras_tuner as kt

# ------------------------------
# Load and preprocess Wine dataset
# ------------------------------
wine = load_wine()
X = wine.data
y = wine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Base model builder
# ------------------------------
def build_model(optimizer='adam', activation='relu', neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X.shape[1], activation=activation))
    model.add(Dense(3, activation='softmax'))
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


optimizers = ['adam', 'sgd', 'rmsprop']
activations = ['relu', 'sigmoid']
neurons_list = [16, 32, 64]
batch_sizes = [16, 32, 64]
epochs_list = [10, 20, 30]

# ------------------------------
# Random Search
# ------------------------------
def random_search(n_iter=5):
    best_acc, best_params = 0, None

    for _ in range(n_iter):
        params = {
            'optimizer': random.choice(optimizers),
            'activation': random.choice(activations),
            'neurons': random.choice(neurons_list),
            'batch_size': random.choice(batch_sizes),
            'epochs': random.choice(epochs_list)
        }

        model = build_model(
            params['optimizer'], params['activation'], params['neurons']
        )

        model.fit(
            X_train, y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            verbose=0
        )

        y_pred = np.argmax(model.predict(X_test), axis=1)
        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc:
            best_acc, best_params = acc, params
    
    return best_params


# ------------------------------
# Grid Search
# ------------------------------
def grid_search():
    best_acc, best_params = 0, None

    for combo in itertools.product(
        optimizers, activations, neurons_list, batch_sizes, epochs_list
    ):
        params = {
            'optimizer': combo[0],
            'activation': combo[1],
            'neurons': combo[2],
            'batch_size': combo[3],
            'epochs': combo[4]
        }

        model = build_model(
            params['optimizer'], params['activation'], params['neurons']
        )

        model.fit(
            X_train, y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            verbose=0
        )

        y_pred = np.argmax(model.predict(X_test), axis=1)
        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc:
            best_acc, best_params = acc, params

    return best_params


# ------------------------------
# Hyperband (Keras Tuner)
# ------------------------------
def hyperband_model(hp):
    model = Sequential()
    model.add(
        Dense(
            hp.Choice('neurons', neurons_list),
            activation=hp.Choice('activation', activations),
            input_shape=(X.shape[1],)
        )
    )
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=hp.Choice('optimizer', optimizers),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


tuner = kt.Hyperband(
    hyperband_model,
    objective='val_accuracy',
    max_epochs=30,
    directory='hyperband_dir',
    project_name='wine'
)

tuner.search(X_train, y_train, validation_split=0.2, verbose=0)

best_hyperband = tuner.get_best_hyperparameters(1)[0].values


# ------------------------------
# Print Results
# ------------------------------
best_random = random_search()
best_grid = grid_search()

print("Best Random Search Params:", best_random)
print("Best Grid Search Params:", best_grid)
print("Best Hyperband Params:", best_hyperband)
