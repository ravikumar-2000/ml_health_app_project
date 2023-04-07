import json
import nltk
import uvicorn
import pandas as pd
from consts import *
from typing import List
from datetime import datetime
from pydantic import BaseModel
from chatbot_test import calling_the_bot

from fastapi import FastAPI, status
from disorder_prediction import predict_model
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentiment_analysis_pretrained import sentiment_vader


try:
    # nltk.download('all')
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
except Exception as e:
    print(e)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MentalDisorderRequestBody(BaseModel):
    feeling_nervous: int
    panic: int
    breathing_rapidly: int
    sweating: int
    trouble_in_concentration: int
    having_trouble_in_sleeping: int
    having_trouble_with_work: int
    hopelessness: int
    anger: int
    over_react: int
    change_in_eating: int
    suicidal_thought: int
    feeling_tired: int
    close_friend: int
    social_media_addiction: int
    weight_gain: int
    material_possessions: int
    introvert: int
    popping_up_stressful_memory: int
    having_nightmares: int
    avoids_people_or_activities: int
    feeling_negative: int
    trouble_concentrating: int
    blamming_yourself: int


class SentimentMessageRequestBody(BaseModel):
    sender_id: str
    message: str
    is_bot: bool
    timestamp: datetime


@app.post("/get-advice")
def get_advice(text: str):
    response = calling_the_bot(text)
    return JSONResponse({"result": response})


@app.post("/get-mental-disorder-status", status_code=status.HTTP_200_OK)
def get_mental_disorder_status(request_body: MentalDisorderRequestBody):
    request_body = request_body.dict()
    print(request_body)
    features = list(request_body.values())
    disorder_status = predict_model(
        mental_disorder_trained_model_filename, labels_filename, features
    )
    return JSONResponse({"result": disorder_status})


@app.post("/get-sentiment-analysis-status", status_code=status.HTTP_200_OK)
def get_sentiment_analysis_report(request_body: List[SentimentMessageRequestBody]):
    final_result = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
    }
    overall_status = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
    }
    for req_body in request_body:
        req_body = req_body.dict()
        if not req_body["is_bot"]:
            scores = sentiment_vader(req_body["message"])
            if scores[-2] >= 0.05:
                overall_status["positive"] += 1
            elif scores[-2] <= -0.05:
                overall_status["negative"] += 1
            else:
                overall_status["neutral"] += 1
    if sum(overall_status.values()) > 0:
        final_result["positive"] = overall_status["positive"] / sum(
            overall_status.values()
        )
        final_result["negative"] = overall_status["negative"] / sum(
            overall_status.values()
        )
        final_result["neutral"] = overall_status["neutral"] / sum(
            overall_status.values()
        )
    return JSONResponse({"result": final_result})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
