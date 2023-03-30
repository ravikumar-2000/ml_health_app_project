import json
import pickle
import pandas as pd
from  consts import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model(
    data_file, le_features, le_target, mental_disorder_trained_model_filename, labels_filename
):
    label_dict = {}
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(".", "_", regex=False)
    for column in df.columns[:-1]:
        df[column] = le_features.fit_transform(df[column])
    df["disorder"] = le_target.fit_transform(df["disorder"])
    for idx, label in enumerate(le_target.classes_):
        label_dict[idx] = label
    json_object = json.dumps(label_dict)
    with open(labels_filename, "w") as outfile:
        outfile.write(json_object)
    df = df.sample(frac=1.0).reset_index(drop=True)
    x_data = df.drop(["disorder"], axis=1).values
    y_data = df["disorder"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=47
    )
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    test_score = rfc.score(x_test, y_test)
    print(f"Test Score: {test_score}")
    pickle.dump(rfc, open(mental_disorder_trained_model_filename, "wb"))
    return x_test, y_test


def predict_model(model_filename, labels_filename, test_features):
    trained_model = pickle.load(open(model_filename, "rb"))
    labels = json.load(open(labels_filename, 'r'))
    prediction = trained_model.predict([test_features])
    return labels[str(prediction[0])]


if __name__ == "__main__":
    le_features = LabelEncoder()
    le_target = LabelEncoder()
    x_test, y_test = train_model(
        data_file_path, le_features, le_target, mental_disorder_trained_model_filename, labels_filename
    )
    predicted_label = predict_model(
        mental_disorder_trained_model_filename,
        labels_filename,
        x_test[5]
    )
    print(predicted_label)
