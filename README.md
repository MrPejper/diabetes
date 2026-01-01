Diabetes Risk Assessment Application
Overview

This application assesses the risk of diabetes based on user-provided health data entered through a simple interface.
The goal of the project is to demonstrate how structured input data, validation logic, and a trained model can be combined into a practical decision-support tool.

The application does not provide a medical diagnosis — it is intended for educational and analytical purposes only.

How It Works:

- The user enters health-related data (e.g. age, BMI, glucose level).

- The application validates the input values.

- If the data is valid, the model evaluates the risk of diabetes.

- The result is returned as a risk assessment.

Input Validation:

The application includes basic input validation to prevent incorrect or unrealistic data, for example:

- Age values outside a reasonable human range (e.g. 200 years)

- Negative or logically invalid health metrics

If invalid data is detected, the application returns a clear error message instead of producing a prediction.

Features:

- User-friendly data input interface

- Input validation for unrealistic values

- Risk classification based on user data

- Clear separation between data validation and prediction logic

Limitations:

- The model is trained on a predefined dataset and may not generalize to all populations

- The application does not replace professional medical advice

- Predictions depend entirely on the quality and accuracy of user-provided data

Technologies Used:

- Python

- Data analysis and preprocessing libraries

- Machine learning model for classification

- Simple application interface for user input

Disclaimer

This project is intended for educational purposes only.
It should not be used as a medical diagnostic tool.
