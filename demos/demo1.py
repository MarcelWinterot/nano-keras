import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from nano_keras.models import NN
from nano_keras.losses import MSE
from nano_keras.layers import Dense, Input
from nano_keras.optimizers import NAdam
from nano_keras.callbacks import EarlyStopping


def load_data() -> tuple:
    X = sns.load_dataset("titanic")

    X = drop_data(X, ["fare", "embarked", "adult_male", "who",
                  "embark_town", "class", "alive", "deck", "alone"])

    y = X.pop("survived")

    encoder = {"male": 0, "female": 1}
    X["sex"] = X["sex"].map(encoder)

    X = X.dropna()
    y = y.loc[X.index]

    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)  # shuffle = False for reproducibility

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = np.array([[y] for y in y_train])
    y_test = np.array([[y] for y in y_test])

    return X_train, X_test, y_train, y_test


def drop_data(data: pd.DataFrame, data_to_drop: list[str]) -> pd.DataFrame:
    for column in data_to_drop:
        data.drop(column, axis=1, inplace=True)
    return data


X_train, X_test, y_train, y_test = load_data()

np.random.seed(1337)

model = NN()
model.add(Input(5))
model.add(Dense(25, "relu"))
model.add(Dense(10, "relu"))
model.add(Dense(5, "relu"))
model.add(Dense(1, "sigmoid"))

optimizer = NAdam()
loss = MSE()

stop = EarlyStopping(5, "val_accuracy", restore_best_weights=True)

model.compile(loss, optimizer, metrics="accuracy")

model.summary()

model.train(X_train, y_train, 50, verbose=2,
            validation_data=(X_test, y_test), callbacks=stop)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\n\nFinished training, the final accuracy: {accuracy}\n\n")
