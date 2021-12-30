# Covid Detection API


<h4 align="center">A REST API that predicts Covid infection from an X-ray image.</h4>

This project is built with FastAPI and the model is trained with TensorFlow, It is a straight-forward project to leverage the FastAPI power and demonstrate it.

## Features

- [x] Cross-platform
- [x] Fast predictions
- [x] No singup/login required

## Code
`prediction_endpoint.py` contains an efficient FastAPI prediction endpoint, it loads the pretrained model and defines the Model class for better usability and readability, it also contains the API logic and exposes a prediction endpoint.

## Built with
- [FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow](https://www.tensorflow.org/api_docs/python/tf)

## Setup
Clone this repo to your desktop or a specific folder you want to run the project on, run `pip install -r requirements.txt` to install all the dependencies.
You might want to create a virtual environment before installing the dependencies.

To run the project on your localhost, you can use `uvicorn prediction_endpoint:app` and it will launch on your localhost.
