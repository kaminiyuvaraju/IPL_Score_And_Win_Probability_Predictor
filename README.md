# IPL Score and Win Probability Prediction

This project aims to predict the scores and win probabilities for Indian Premier League (IPL) matches using machine learning techniques. The goal is to build a predictive model that can estimate the score of a match in progress and calculate the probability of a team winning based on real-time data inputs such as match statistics, historical data, and player performance.

## Features

- **Score Prediction**: Predict the final score of a team based on current match data.
- **Win Probability**: Calculate the probability of a team winning the match given the current context.
- **Real-Time Prediction**: Predict scores and win probability in real-time during a live match.
- **Historical Data Analysis**: Utilize past IPL data for training the model and improving accuracy.

## Technologies Used

- **Python**: The main programming language used.
- **Machine Learning Libraries**: 
  - `scikit-learn` for building prediction models
- **Pandas** for data manipulation and preprocessing
- **NumPy** for numerical computations
- **Matplotlib / Seaborn** for data visualization
- **Flask**: For deploying the model as a web application

## Algorithms Used

1. **Linear Regression**: A linear approach to model the relationship between match statistics and the predicted score.
2. **Random Forest**: An ensemble method that builds multiple decision trees and aggregates their predictions to improve accuracy.
3. **Gradient Boosting**: A boosting method that combines weak learners to improve predictive performance by focusing on errors from previous iterations.

## Installation and Running the App

### Step 1: Clone, Install, and Run

Follow the steps below to set up and run the project:

```bash
# Clone the repository
git clone https://github.com/your-username/ipl-score-win-prediction.git

# Navigate to the project directory
cd ipl-score-win-prediction

# Install the required libraries
pip install -r requirements.txt

# Start the Flask server
python app.py
# Once the server is running, open your browser and visit:

http://127.0.0.1:5000/
