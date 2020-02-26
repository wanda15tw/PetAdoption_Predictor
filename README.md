# PetAdoption_Predictor

## Introduction

This project is inspired by Kaggle competition: [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction/) and the data sources are solely from there. 

This repository mainly consists of `src` and `flaskapp`, and excluding raw data, intermediate and processed data, etc.
* `src`: source codes of all data pipelines
* `flaskapp`: source codes to host a simple app to make a new pet's adoption speed

## `SRC`
* `d01_data.read_data.py`: read provided `.csv` files into Pandas DataFr
* `d02_intermeidate.preprocessing.py`: preprocessing
* `d03_processing.build_features.py`: processing to build features for modeling
* `d04_modeling.train_model.py`: training & model_selection
* `d04_modeling.predict_model.py`: 

## `flaskapp`

* Under the directory, run `python app.py` to start a simple flask app
* Open browser and navigate to `localhost:5000` to *GET* the input page
* Input pet information and hit *submit* to *POST*
* Wait while the programming to process the input entries and make prediction based on the saved model
* See the prediction result!