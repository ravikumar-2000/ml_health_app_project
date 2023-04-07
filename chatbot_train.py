import json
import pickle
import random

import nltk
import numpy as np
from consts import *
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

try:
    nltk.download('all')
except Exception as e:
    print(e)


def process_intents_create_training_dataset():

    words = []
    classes = []
    documents = []
    dataset = []

    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open(intents_file_path).read())

    ignore_letters = ["?", "!", ".", ","]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))

            if intent["tag"] not in classes:
                classes.append(intent["tag"])
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

    words = sorted(set(words))
    classes = sorted(set(classes))

    pickle.dump(words, open(inetnts_words_file, "wb"))
    pickle.dump(classes, open(inetnts_classes_file, "wb"))

    template = [0] * len(classes)
    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(template)
        output_row[classes.index(document[1])] = 1
        dataset.append([bag, output_row])

    random.shuffle(dataset)
    dataset = np.array(dataset)

    train_x = list(dataset[:, 0])
    train_y = list(dataset[:, 1])

    return train_x, train_y


def create_model(input_data_length, output_data_length):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_data_length,), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_data_length, activation="softmax"))

    sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    return model


def train_model(x_train, y_train, model):
    hist = model.fit(
        np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1
    )
    model.save(chatbot_model_path, hist)
    print("trained model saved!")


if __name__ == "__main__":
    x_train, y_train = process_intents_create_training_dataset()
    model = create_model(len(x_train[0]), len(y_train[0]))
    train_model(x_train, y_train, model)
