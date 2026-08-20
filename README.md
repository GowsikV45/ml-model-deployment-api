# ML Model Deployment as a Monitored REST API

## Project Overview

This project builds a REST API that uses a machine learning model to classify Iris flowers.

The API will accept four flower measurements as input and return the predicted Iris flower species.

## Machine Learning Problem

The problem is a classification task.

The model will classify an Iris flower into one of three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

## Dataset

The project will use the Iris dataset provided by scikit-learn.

The dataset contains four input features:

- Sepal length
- Sepal width
- Petal length
- Petal width

## API Contract

The `/predict` endpoint will accept the four flower measurements as input.

Example input:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
