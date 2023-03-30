from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_vader(sentence):

    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict["neg"]
    neutral = sentiment_dict["neu"]
    positive = sentiment_dict["pos"]
    compound = sentiment_dict["compound"]

    if sentiment_dict["compound"] >= 0.05:
        overall_sentiment = "Positive"

    elif sentiment_dict["compound"] <= -0.05:
        overall_sentiment = "Negative"

    else:
        overall_sentiment = "Neutral"

    print(negative, neutral, positive, compound, overall_sentiment)
    return [negative, neutral, positive, compound, overall_sentiment]


if __name__ == "__main__":
    print(sentiment_vader("hey i am realy happy"))
    print(sentiment_vader("this is the worst day of my entire life"))
