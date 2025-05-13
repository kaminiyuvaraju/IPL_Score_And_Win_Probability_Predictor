from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import random
import logging

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load models and encoders
score_model = joblib.load('model/ipl_score_predictor_model.pkl')
win_model = joblib.load('model/win_predictor.pkl')
team_encoder = joblib.load('model/team_encoder.pkl')
venue_encoder = joblib.load('model/venue_encoder.pkl')

# Prepare clean list of teams and venues
raw_teams = list(team_encoder.classes_)
filtered_teams = [team for team in raw_teams if team != "Kochi Tuskers Kerala"]

final_teams = []
seen = set()
for team in filtered_teams:
    norm = team.lower().strip()
    if "royal challengers" in norm:
        if "royal challengers bangalore" not in seen:
            final_teams.append("Royal Challengers Bangalore")
            seen.add("royal challengers bangalore")
    elif norm not in seen:
        final_teams.append(team)
        seen.add(norm)

teams = sorted(final_teams)
venues = sorted(list(venue_encoder.classes_))

@app.route('/')
def index():
    return render_template('index.html', teams=teams, venues=venues)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        venue = request.form['venue']
        total_runs = int(request.form['total_runs'])
        is_wicket = int(request.form['is_wicket'])
        overs = float(request.form['overs'])      # ✅ Accept decimal overs like 9.8
        balls = int(request.form['balls'])         # ✅ Use float() if fractional input is allowed

        # Encode categorical variables
        batting_team_enc = team_encoder.transform([batting_team])[0]
        bowling_team_enc = team_encoder.transform([bowling_team])[0]
        venue_enc = venue_encoder.transform([venue])[0]

        # 🎯 Score Prediction
        score_input = np.array([[total_runs, is_wicket, batting_team_enc, bowling_team_enc, venue_enc]])
        predicted_score = int(round(score_model.predict(score_input)[0]))
        score_range = f"{predicted_score - 5} - {predicted_score + 5}"

        # 🎯 Win Probability Prediction using 7 features
        win_input = np.array([[batting_team_enc, bowling_team_enc, venue_enc,
                               total_runs, is_wicket, overs, balls]])

        logging.info(f"Win model input: {win_input}")

        if hasattr(win_model, "predict_proba"):
            win_prob = win_model.predict_proba(win_input)[0][1]
        else:
            win_prob = float(win_model.predict(win_input)[0])
            win_prob = max(0.0, min(win_prob, 1.0))

        win_prob_display = f"{win_prob * 100:.2f}%"
        logging.info(f"Win probability: {win_prob_display}")

        # 📈 Match Progress Chart
        over_list = list(range(1, 6))
        runs_per_over = [random.randint(5, 10) for _ in over_list]
        wickets_per_over = [random.choice([0, 0, 1]) for _ in over_list]

        fig, ax1 = plt.subplots()
        ax1.plot(over_list, runs_per_over, 'g-o', label="Runs per Over")
        ax1.set_xlabel("Overs")
        ax1.set_ylabel("Runs", color='g')

        for i, w in enumerate(wickets_per_over):
            if w > 0:
                ax1.plot(over_list[i], runs_per_over[i], 'ro', label='Wicket' if i == 0 else "")

        plt.title("Match Progression - First 5 Overs")
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graph_url = base64.b64encode(buf.getvalue()).decode('utf8')
        buf.close()
        plt.close(fig)

        return render_template(
            'result.html',
            prediction=score_range,
            batting_team=batting_team,
            bowling_team=bowling_team,
            venue=venue,
            total_runs=total_runs,
            is_wicket=is_wicket,
            overs=overs,
            graph=graph_url,
            win_prob=win_prob_display
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
