from typing import Literal, List, Optional
from collections import Counter
from functools import partial
import re
import string
import pandas as pd
import numpy as np
import dill

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


MODEL_PATH = r"/opt/app/data/models/test_20240712.joblib"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        loaded_pipeline = dill.load(f)
    return loaded_pipeline


def standartize_columns(df):
    new_columns = ["_".join(x.lower().split(" ")) for x in list(df.columns)]
    df.columns = new_columns
    return df


def text_cleaning(df, columns: List[str]):
    for column in columns:
        # lower
        df[column] = df[column].apply(lambda x: x.lower())

        # Remove digits and words containing digits
        # df[column]=df[column].apply(lambda x: re.sub('\w*\d\w*','', x))

        # Remove Punctuations
        df[column] = df[column].apply(lambda x: re.sub("[%s]" % re.escape(string.punctuation), "", x))

        # Removing extra spaces
        df[column] = df[column].apply(lambda x: re.sub(" +", " ", x))
    return df


def standartize_text(df, columns: List[str]):
    def func(text):
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text

    for column in columns:
        df[column] = df[column].apply(func)
    return df


def word_count(text: str) -> int:
    text_list = text.split()
    return len(text_list)


def get_top_values(text: str, n: int = 10):
    return " ".join([val for val, count in Counter(text.split(" ")).most_common(n)])


class ProcessingStep:

    def __init__(self, context: Literal["train", "inference"]):
        self.context = context

    @staticmethod
    def data_cleaning(df: pd.DataFrame, context: str) -> pd.DataFrame:
        # text cleaning
        df = text_cleaning(df=df, columns=["title", "abstract"])
        df = standartize_text(df=df, columns=["title", "abstract"])

        # concatinating title and abstract. We'll treat it as one
        # as some abstracts are uninformative, however title is
        df["text"] = df["title"] + " " + df["abstract"]
        df = df.drop(["title", "abstract"], axis=1)
        if context == "inference":
            return df

        # Concatenate values into 'label' column
        label_columns = [
            "computer_science",
            "physics",
            "mathematics",
            "statistics",
            "quantitative_biology",
            "quantitative_finance",
        ]
        df["label"] = df.apply(lambda row: "|".join([col for col in label_columns if row[col] == 1]), axis=1)
        df = df.drop(label_columns, axis=1)

        return df

    @staticmethod
    def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        df["length"] = df["text"].str.len()
        df["word_count"] = df["text"].apply(word_count)
        df["mean_word_length"] = df["text"].map(lambda rev: np.mean([len(word) for word in rev.split()]))
        df["text_top_5"] = df["text"].apply(partial(get_top_values, n=5))
        df["text_top_10"] = df["text"].apply(partial(get_top_values, n=10))
        return df

    @staticmethod
    def deduplication(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(df[df.duplicated("text", keep="first")].index)
        return df

    @staticmethod
    def fixing_cardinality(df: pd.DataFrame) -> pd.DataFrame:
        cardinality_treshold = len(df) * 0.02
        labels_to_change = [val for val, count in Counter(df["label"]).items() if count > cardinality_treshold]
        df["label"] = df["label"].apply(lambda x: x if x in labels_to_change else "other")
        return df

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = standartize_columns(df=df)
        df = self.data_cleaning(df=df, context=self.context)
        df = self.feature_engineering(df=df)
        if self.context == "inference":
            return df
        df = self.deduplication(df=df)
        df = self.fixing_cardinality(df=df)

        # fixing indexing
        df.reset_index(drop=True, inplace=True)
        return df


MODEL = load_model()


def predict(raw_data) -> str:
    global MODEL
    raw_df = pd.DataFrame([raw_data], columns=["title", "abstract"])
    df = ProcessingStep(context="inference").execute(df=raw_df)
    out = MODEL.predict(pd.concat([df, df]))
    return out[0]


app = FastAPI()
# Define CORS settings
origins = ["*"]  # Allow requests from any origin

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class Item(BaseModel):
    title: str
    abstract: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Real estate valuation using machine learning techniques",
                    "abstract": "Finally, the results of the best model with 13.74 MAPE and 33,307 RMSE are presented along with the conclusions of the work.",
                }
            ]
        }
    }


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/")
async def predict_api(item: Item):
    result = predict(item.dict())
    return {"status": 200, "result": result}
