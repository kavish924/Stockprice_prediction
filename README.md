# Stockprice_prediction

This project uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices, specifically for Apple Inc. (AAPL). The model is built with TensorFlow and Keras and includes data preprocessing, feature engineering, and detailed evaluation.

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Setup and Installation](#-setup-and-installation)
- [How to Run](#-how-to-run)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [License](#-license)
- [Contributing](#-contributing)

---

## 🚀 Project Overview

The primary goal of this project is to build a robust time-series forecasting model for stock prices. It leverages historical stock data to train an LSTM network, a type of recurrent neural network (RNN) well-suited for sequence prediction problems. The notebook demonstrates an end-to-end workflow, from loading and cleaning the data to building, training, and evaluating the predictive model.

## ✨ Features

- **Data Preprocessing**: Cleans and formats raw stock data for time-series analysis.
- **Feature Engineering**: Creates additional technical indicators to improve model performance, including:
    - Moving Averages (MA): 5-day, 10-day, and 20-day
    - Relative Strength Index (RSI)
    - Bollinger Bands (Upper and Lower)
    - Percentage Price Change
    - Volume Moving Average
    - High-Low Spread
- **Advanced Model Training**:
    - Implements `EarlyStopping` to prevent overfitting.
    - Uses `ReduceLROnPlateau` to adjust the learning rate dynamically.
- **Comprehensive Evaluation**: Assesses model performance using a variety of metrics:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)
    - R-squared (R²)

## 📊 Dataset

The model is trained on historical stock data for **Apple Inc. (AAPL)** from January 2016 to December 2023. The dataset (`AAPL_stock_data.csv`) should contain the following columns:
- `Date`
- `Close`
- `High`
- `Low`
- `Open`
- `Volume`

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow & Keras**: For building and training the LSTM model.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For data scaling and evaluation metrics.
- **google colab**: For interactive development and documentation.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
    pandas
    numpy
    matplotlib
    scikit-learn
    tensorflow
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

1.  Place your dataset file (e.g., `AAPL_stock_data.csv`) in the root directory of the project.
2.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open the `stockprice_prediciton.ipynb` file and run the cells sequentially.

## 🧠 Model Architecture

The LSTM model is constructed as a `Sequential` model in Keras with the following layers:

1.  **LSTM Layer 1**: 50 units, `return_sequences=True`
2.  **Dropout**: 20%
3.  **LSTM Layer 2**: 50 units, `return_sequences=True`
4.  **Dropout**: 20%
5.  **LSTM Layer 3**: 50 units, `return_sequences=False`
6.  **Dropout**: 20%
7.  **Dense Layer**: 25 units, ReLU activation
8.  **Dropout**: 10%
9.  **Output Dense Layer**: 1 unit

The model is compiled with the `Adam` optimizer and `Mean Squared Error` as the loss function.

## 📈 Results

The model was evaluated on the test set, yielding the following performance metrics:

| Metric                      | Value        |
| --------------------------- | ------------ |
| Mean Squared Error (MSE)    | 0.0096       |
| Mean Absolute Error (MAE)   | 0.0787       |
| Root Mean Squared Error(RMSE) | 0.0977       |
| MAPE (%)                    | 9.06%        |
| **Accuracy (100-MAPE) (%)** | **90.94%** |
| R-squared (R²)              | 0.1868       |

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, please feel free to create a pull request or open an issue.
