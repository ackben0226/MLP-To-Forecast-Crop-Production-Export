# Crop Price Forecasting with Multilayer Perceptron (MLP)
This repository provides a machine learning model that leverages a Multilayer Perceptron (MLP) to forecast the price of food crops in a specific geographic region over the next three years. The model uses historical crop production, price, and regional data to predict future prices, assisting stakeholders in decision-making and price trend analysis.

- Table of Contents
- Project Overview
- Data
- Project Structure
- Requirements
- Usage
- Model Training and Prediction
- Results
- Future Improvements
- Contributing
- License

## Project Overview
This project aims to create a robust forecasting model for food crop prices using an MLP neural network. The MLP model processes historical crop prices and production-related data to predict future prices. This forecast can support policymakers, farmers, and supply chain managers in planning and preparing for market trends.

## Data
The model requires historical data for the target crop, including:

Yearly production levels
Price per unit of crop
Area and domain (e.g., specific geographic identifiers)
This dataset should span multiple years to capture trends and seasonality in crop pricing.

## Data Preparation
The dataset is cleaned and transformed using a preprocessing pipeline.
The most common values for categorical variables, such as Area and Domain, are used for consistency in predictions for future years.
The processed data is then split into training and test sets for model evaluation.
Project Structure
The project files are organized as follows:

## Multilayer_Perceptron_To_Forecast_Crop_Production_Export.ipynb: 
The main notebook file contains data preprocessing, model training, evaluation, and prediction steps.
README.md: Project documentation.
requirements.txt: List of required Python packages for easy setup.
Requirements

## To set up the project environment, install the required libraries using:

bash
Copy code
pip install -r requirements.txt
Key dependencies include:

tensorflow
pandas
numpy
matplotlib
scikit-learn
Usage
Clone the repository:

bash

Copy code

git clone https://github.com/ackben0226/MLP-To-Forecast-Crop-Production-Export.git
Run the Jupyter Notebook: Open Multilayer_Perceptron_To_Forecast_Crop_Production_Export.ipynb to follow the step-by-step guide for loading the data, preprocessing, training the MLP model, and making predictions.

## Model Training and Prediction
Data Transformation: A preprocessing pipeline standardizes the input features.
</ Model Architecture: The MLP model architecture is defined with layers to suit the data complexity and achieve accurate predictions.
Training: Model is trained with a specified optimizer and loss function, with performance tracked through metrics.
Prediction: Each year, future crop prices are predicted within the forecast horizon. Predictions are output to a DataFrame and visualized as a time-series trend plot.
Results
The output includes:

## Predicted Crop Prices: A table showing projected prices over the forecasted years.
Visualization: A plot displaying the predicted crop prices across the forecast period.
The results provide insights into the expected price trajectory for the selected crop, helping users make informed decisions based on forecasted trends.

## Future Improvements
Consider the following for enhanced forecasting accuracy:

Additional Data: Incorporate more features (e.g., weather patterns, market demand) to improve predictive power.
Advanced Models: Experiment with more complex models like LSTM or GRU, which may capture time-dependent patterns better.
Model Tuning: Fine-tune hyperparameters for optimal model performance.
Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request for review.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.
