import json
import pickle
import random
import time

import nltk
import numpy as np
from consts import *
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()
intents = json.loads(open(intents_file_path).read())

words = pickle.load(open(inetnts_words_file, "rb"))
classes = pickle.load(open(inetnts_classes_file, "rb"))
model = load_model(chatbot_model_path)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list) > 0:
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]

        result = ""

        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
        return result
    return None


def calling_the_bot(text):
    res = None
    predict = predict_class(text)
    res = get_response(predict, intents)
    print("Your Symptom was : ", text)
    print("Result found in our Database : ", res)
    return res


if __name__ == "__main__":

    print("Bot is Running")

    while True:
        text = input("tell us your symptoms => ")

        if len(text) > 0:
            calling_the_bot(text)
