# 🚕 Taxi Trip Price Prediction Using Polynomial Regression

An end-to-end Machine Learning project that predicts taxi trip fares based on trip details such as distance, duration, passenger count, time of day, traffic conditions, weather, and pricing parameters.

The project combines a Polynomial Regression model with a Streamlit web application and an interactive map-based location experience using the LocationIQ API.

---

## 📌 Project Overview

Taxi fares can vary depending on several factors such as:

- Trip distance
- Trip duration
- Passenger count
- Time of day
- Day of week
- Traffic conditions
- Weather
- Base fare
- Per-kilometer rate
- Per-minute rate

This project uses Machine Learning to estimate the expected taxi trip price from these factors.

The trained Polynomial Regression model is deployed through a Streamlit application where users can select pickup and drop locations, automatically calculate route information, enter trip details, and receive an estimated fare.

---

## 🎯 Project Objective

The main objectives of this project are:

1. Build a Machine Learning model to predict taxi trip prices.
2. Compare the relationship between trip features and the target price.
3. Use Polynomial Regression to capture nonlinear relationships and feature interactions.
4. Develop an interactive Streamlit web application.
5. Integrate live location search using the LocationIQ API.
6. Calculate road distance and estimated travel duration.
7. Display pickup and drop locations on an interactive map.
8. Provide an instant estimated taxi fare.

---

## 📊 Dataset

The project uses the `taxi_trip_pricing.csv` dataset.

### Dataset Information

- **Records:** 1,000
- **Total Columns:** 11
- **Categorical Features:** 4
- **Target Variable:** `Trip_Price`

### Features

| Feature | Description |
|---|---|
| `Trip_Distance_km` | Distance travelled during the trip |
| `Time_of_Day` | Morning, Afternoon, Evening, or Night |
| `Day_of_Week` | Day on which the trip occurred |
| `Passenger_Count` | Number of passengers |
| `Traffic_Conditions` | Low, Medium, or High |
| `Weather` | Clear, Rain, or Cloudy |
| `Base_Fare` | Base taxi fare |
| `Per_Km_Rate` | Fare charged per kilometer |
| `Per_Minute_Rate` | Fare charged per minute |
| `Trip_Duration_Minutes` | Trip duration |
| `Trip_Price` | Target variable |

---

## 🔎 Exploratory Data Analysis

The project includes exploratory analysis to understand the dataset and relationships between variables.

The analysis includes:

- Dataset inspection
- Data types
- Missing-value analysis
- Descriptive statistics
- Unique-value analysis
- Target-variable distribution
- Distance vs. price analysis
- Duration vs. price analysis
- Categorical feature analysis
- Box plots
- Correlation analysis
- Data visualization using Matplotlib and Seaborn

---

## 🧹 Data Preprocessing

The following preprocessing steps are used before model training:

### 1. Handle Missing Values

Missing numerical values are handled using the column mean.

### 2. Label Encoding

`Day_of_Week` is converted into numerical values using `LabelEncoder`.

### 3. One-Hot Encoding

Categorical features such as:

- `Time_of_Day`
- `Traffic_Conditions`
- `Weather`

are converted into numerical features using one-hot encoding.

### 4. Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

with:

```python
random_state=42
