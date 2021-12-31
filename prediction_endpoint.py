"""Prediction endpoint """
import typing as t
from pathlib import Path
import secrets
import logging

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image


logging.basicConfig(
    filename="predictions.log", level=logging.INFO, format="%(asctime)s | %(message)s"
)
MODEL_PATH = Path("./models/CovidDetector.h5")
IMG_DIR = Path("./images")


class PredictionInput(BaseModel):
    image: bytes


class PredictionOutput(BaseModel):
    isInfected: str


class CovidDetectionModel:
    """
    A class used to represent a Model.

    Attributes:
        model (tf.keras.Model) : Our pretrained prediction model.

    Methods:
        load_model()
            Loads the pretrained model to a variable.

        preprocess(image : PIL.Image) (staticmethod)
            Preprocess our image for model prediction. Our model accepts a 299 x 299 image.

        predict(image : bytes)
            Reads the image from the request and predicts the class of the image.
    """

    model: t.Optional[tf.keras.Model]

    def load_model(self):
        """Loads the model."""
        self.model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("The model was loaded to memory.")

    @staticmethod
    def preprocess(image: Image):
        """Preprocess the input image."""
        return tf.image.resize_with_crop_or_pad(image, 299, 299)

    @staticmethod
    def save_image(image: Image, extenstion: str):
        """Saves the received image, and returns the saved image path."""
        image_file = secrets.token_hex(8) + "." + extenstion
        saving_path = Path(str(IMG_DIR) + "/" + image_file)
        image.save(saving_path)
        return saving_path

    async def predict(
        self,
        image: UploadFile = File(...),
    ) -> PredictionOutput:
        """Runs a prediction."""
        if not self.model:  # If we fail to load the model we raise an error.
            logging.error("The model was not loaded successfully to memory.")
            raise RuntimeError("Model is not loaded.")
        if "image" not in image.content_type.split(
            "/"
        ):  # Check if the uploaded image is in a suported format.
            logging.error("None supported file type was sent to the API.")
            raise HTTPException(400, detail="The file passed is not an image.")
        data = Image.open(image.file)
        image_path = CovidDetectionModel.save_image(
            data, image.content_type.split("/")[-1]
        )
        img = tf.keras.preprocessing.image.array_to_img(data)
        prediction = self.model.predict(
            np.expand_dims(CovidDetectionModel.preprocess(img), axis=0)
        )
        label = prediction.argmax()
        if label:
            isInfected = "True"
        else:
            isInfected = "False"
        logging.info(f"{image_path} | model prediction : {isInfected}")
        return PredictionOutput(isInfected=isInfected)


# Initializing the API.
app = FastAPI(
    title="Covid Detection API.",
    description="""Obtain a prediction for covid infection from a chest X-ray image.""",
    version="0.1.0",
)
covid_model = CovidDetectionModel()

# Adding the prediction endpoint to the API.
@app.post("/prediction")
async def prediction(
    isInfected: PredictionOutput = Depends(covid_model.predict),
) -> PredictionOutput:
    return isInfected


# Loading the model to memory on program startup.
@app.on_event("startup")
async def startup():
    """Load the persisted model on startup up to speed up the process."""
    covid_model.load_model()
