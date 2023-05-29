# Titanic Survival Prediction
This project aims to predict the survival of passengers on the Titanic Disaster Dataset from Kaggle. Instead of focusing on deploying complicated models, the main goal of this project is to provide clean and maintainable code.

## Project Overview
In this project, we have implemented three different classification models: a simple DNN (Deep Neural Network), a Random Forest, and an XGBoost classifier. By utilizing a bagging strategy, we combine the predictions from these three models to generate more accurate results. The achieved accuracy is around 78%.

## Usage

1. Install the required packages by running the following command:
```console
pip install -r src\requirements.txt
```

2. Train the models and obtain the predicted results by running the following command:
```console
python3 src\main.py
```

## Author 
This project was developed by Kai-Hsin Chen.