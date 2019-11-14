import os

import tensorflow as tf
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def etl(csv_path, batch_size):
    data_frame = pd.read_csv(csv_path)

    data_frame.drop("PassengerId", axis=1, inplace=True)
    data_frame.drop("Name", axis=1, inplace=True)
    data_frame.drop("Ticket", axis=1, inplace=True)
    data_frame.drop("Cabin", axis=1, inplace=True)

    mean = data_frame.mean(axis=0, skipna=True)
    data_frame.fillna(mean, inplace=True)

    data_frame["Sex"] = pd.Categorical(data_frame["Sex"])
    data_frame["Sex"] = data_frame.Sex.cat.codes

    data_frame["Embarked"] = pd.Categorical(data_frame["Embarked"])
    data_frame["Embarked"] = data_frame.Embarked.cat.codes

    numerical_data = data_frame[["Age", "SibSp", "Parch", "Fare"]]
    numerical_data = (numerical_data - numerical_data.min()) / (numerical_data.max() - numerical_data.min())

    data_frame[["Age", "SibSp", "Parch", "Fare"]] = numerical_data

    if csv_path == "train.csv":
        train_labels = data_frame.pop("Survived")
        data = tf.data.Dataset.from_tensor_slices((dict(data_frame), train_labels))
        data = data.batch(batch_size)
        return data

    return data_frame.to_numpy()


def input_layer():
    categories = {
        'Sex': [1, 0],
        'Embarked': [0, 1, 2],
        "Pclass": [1, 2, 3]
    }

    categorical_columns = []
    for feature, vocab in categories.items():
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(categorical_column))

    return tf.keras.layers.DenseFeatures(categorical_columns)


if __name__ == '__main__':
    train_data = etl("train.csv", 32)

    model = tf.keras.Sequential([
        input_layer(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss=tf.losses.mean_squared_error,
        optimizer="adam",
        metrics=['accuracy', 'Precision', 'Recall'])

    model.fit(train_data, epochs=10)

    test_data = etl("test.csv", 2)
    test_data_sorted = [test_data[:, 2],
                        test_data[:, 6].astype(int),
                        test_data[:, 5],
                        test_data[:, 4],
                        test_data[:, 0].astype(int),
                        test_data[:, 1].astype(int),
                        test_data[:, 3]]
    predictions = model.predict(test_data_sorted)

    index = 0
    for prediction in predictions:
        sex = "Masculino" if test_data[index][1] == 1.0 else "Feminino"
        print("Sex: " + sex + " | " + str(prediction))
        index = index + 1
