"""
Keystroke Dynamics Data Collection and Analysis Script
------------------------------------------------------

This script allows you to collect and analyze keystroke dynamics data for multiple users.
It supports four modes of operation: 'collect', 'analyze', 'both', and 'test'.

Modes:
- collect: Collect keystroke data for multiple users and save it to a CSV file.
- analyze: Analyze the collected keystroke data, train a machine learning model, and predict user identity.
- both: Perform both data collection and analysis.
- test: Test the user identity prediction using an existing model.

Usage:
- Collect data: 
  python3 biometric-keystrokes.py --mode collect --filename keystroke_data.csv --users 2
- Analyze data: 
  python3 biometric-keystrokes.py --mode analyze --filename keystroke_data.csv
- Collect and analyze data: 
  python3 biometric-keystrokes.py --mode both --filename keystroke_data.csv --users 2
- Test user identity: 
  python3 biometric-keystrokes.py --mode test --stats_file stats.json --model_file model_and_encoders.pkl

Files:
- keystroke_data.csv: CSV file to store collected keystroke data.
- model_and_encoders.pkl: File to save the trained model and encoders.
- stats.json: JSON file to store prediction statistics.

Options:
- --mode: Mode to run the script in: collect, analyze, both, or test.
- --filename: Filename for keystroke data CSV.
- --stats_file: Filename for statistics JSON.
- --users: Number of users for data collection.

Additional Options:
- --clear_stats: Clear the statistics file.
- --clear_model: Clear the model and encoders file.

For more information, please refer to the script documentation or run the script with the --help option.

"""

import keyboard
import time
import csv
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse
import os
import json
import joblib

# Function to collect keystroke data for a specified user
def collect_keystroke_data_for_user(user_name, stop_key='esc'):
    print(f"Recording keystroke data for user '{user_name}'. Press '{stop_key}' to stop.")
    data = []
    last_press_time = None
    key_down_times = {}
    time_since_last_press = 0

    def on_event(event):
        nonlocal last_press_time, time_since_last_press
        current_time = event.time

        if event.event_type == 'down':
            time_since_last_press = (current_time - last_press_time) if last_press_time else None
            key_down_times[event.name] = current_time
            last_press_time = current_time

        elif event.event_type == 'up' and event.name in key_down_times:
            press_time = key_down_times.pop(event.name)
            time_between_press_release = current_time - press_time
            data.append([user_name, event.name, press_time, time_between_press_release, time_since_last_press])
            if time_since_last_press is not None:
                print(f"Recorded: User={user_name}, Key={event.name}, PressTime={press_time:.4f}, HoldTime={time_between_press_release:.4f}, TimeSinceLastPress={time_since_last_press:.4f}")
            else:
                print(f"Recorded: User={user_name}, Key={event.name}, PressTime={press_time:.4f}, HoldTime={time_between_press_release:.4f}, TimeSinceLastPress=None")

    keyboard.hook(on_event)

    try:
        while True:
            if keyboard.is_pressed(stop_key):
                print("Recording stopped.")
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()

    return data

# Function to save data to CSV
def save_data_to_csv(data, filename="keystroke_data.csv"):
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["User", "Key", "Press Timestamp", "Hold Time", "Time Since Previous Key Press"])
        writer.writerows(data)
    print(f"Keystroke data saved to '{filename}'.")

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, encoder=None, scaler=None, return_encoders=True):
    print("Columns in the dataset:", data.columns)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore')
        key_events = encoder.fit_transform(data[['Key']]).toarray()
    else:
        key_events = encoder.transform(data[['Key']]).toarray()

    if scaler is None:
        scaler = StandardScaler()
        temporal_features = scaler.fit_transform(data[['Press Timestamp', 'Hold Time', 'Time Since Previous Key Press']])
    else:
        temporal_features = scaler.transform(data[['Press Timestamp', 'Hold Time', 'Time Since Previous Key Press']])

    features = np.hstack((key_events, temporal_features))

    if return_encoders:
        user_encoder = OneHotEncoder(handle_unknown='ignore')
        user_labels = data['User']
        user_labels_encoded = user_encoder.fit_transform(user_labels.values.reshape(-1, 1)).toarray()
        return features, user_labels, user_labels_encoded, user_encoder, encoder, scaler
    else:
        return features

# Function to test multiple classifiers with GridSearchCV for hyperparameter tuning
def test_classifiers(features, labels_encoded):
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    param_grids = {
        'Decision Tree': {'max_depth': [None, 10, 20, 30]},
        'Random Forest': {'n_estimators': [10, 50, 100]},
        'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.01]},
        'Logistic Regression': {'C': [0.1, 1, 10]}
    }

    best_model = None
    best_accuracy = 0
    best_classifier_name = ""
    results = {}

    for name, clf in classifiers.items():
        grid_search = GridSearchCV(clf, param_grids[name], cv=5)
        X_train, X_test, y_train, y_test = train_test_split(features, np.argmax(labels_encoded, axis=1), test_size=0.25, random_state=42)
        grid_search.fit(X_train, y_train)
        accuracy = grid_search.score(X_test, y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")

        results[name] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = grid_search.best_estimator_
            best_classifier_name = name

    print(f"Best Classifier: {best_classifier_name} with Accuracy: {best_accuracy:.4f}")

    return best_model, best_classifier_name, best_accuracy, results

# Collect keystroke data for multiple users
def collect_keystroke_data_for_multiple_users(num_users, filename):
    all_data = []
    user_names = []
    for user_id in range(num_users):
        while True:
            user_name = input(f"Enter name for user {user_id+1} (single word): ").strip()
            if ' ' not in user_name:
                break
            print("Error: Name should be a single word without spaces. Please try again.")
        user_names.append(user_name)
        data = collect_keystroke_data_for_user(user_name)
        all_data.extend(data)
    save_data_to_csv(all_data, filename)

# Analyze data
def analyze_data(file_path):
    data = load_data(file_path)

    features, user_labels, user_labels_encoded, user_encoder, encoder, scaler = preprocess_data(data)

    model, best_classifier_name, best_accuracy, results = test_classifiers(features, user_labels_encoded)

    joblib.dump((model, user_encoder, encoder, scaler, results), 'model_and_encoders.pkl')

    return model, user_encoder, encoder, scaler, results

# Function to load model and encoders
def load_model_and_encoders():
    if os.path.exists('model_and_encoders.pkl'):
        model, user_encoder, encoder, scaler, results = joblib.load('model_and_encoders.pkl')
        return model, user_encoder, encoder, scaler, results
    else:
        print("No existing model found. Please run in 'analyze' mode to create a model.")
        return None, None, None, None, None

# Function to predict and validate a user's identity
def predict_and_validate_user(model, user_encoder, encoder, scaler, stats_file):
    print("Press Enter to start typing as the test user.")
    input()
    print("Type some text to identify the user. Press 'esc' to stop.")
    while True:
        data = collect_keystroke_data_for_user('test')
        if len(data) < 5:
            print("Not enough data collected. Please type more and try again.")
        else:
            break

    features = preprocess_data(pd.DataFrame(data, columns=["User", "Key", "Press Timestamp", "Hold Time", "Time Since Previous Key Press"]), encoder, scaler, return_encoders=False)

    if len(features) == 0:
        print("Not enough data collected. Please try again.")
        return False

    predictions = model.predict_proba(features)
    predicted_user_indices = np.argmax(predictions, axis=1)

    unique_labels = user_encoder.categories_[0]
    predicted_user_names = unique_labels[predicted_user_indices]

    predicted_user_name, counts = np.unique(predicted_user_names, return_counts=True)
    predicted_user_name = predicted_user_name[np.argmax(counts)]
    
    print(f"Predicted User: {predicted_user_name}")

    while True:
        feedback = input("Was the prediction correct? (yes/no): ").strip().lower()
        if feedback in ['yes', 'no']:
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    
    record_stats(stats_file, predicted_user_name, feedback == 'yes')
    return feedback == 'yes'

# Function to record statistics
def record_stats(stats_file, predicted_user, correct):
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as file:
            stats = json.load(file)
    else:
        stats = {'total': 0, 'correct': 0}

    stats['total'] += 1
    if correct:
        stats['correct'] += 1

    with open(stats_file, 'w') as file:
        json.dump(stats, file)

    print(f"Current accuracy: {stats['correct'] / stats['total']:.2f}")

# Function to clear statistics
def clear_stats(stats_file):
    if os.path.exists(stats_file):
        os.remove(stats_file)
        print(f"Statistics file '{stats_file}' cleared.")
    else:
        print(f"Statistics file '{stats_file}' does not exist.")

# Function to clear model and encoders
def clear_model_and_encoders():
    if os.path.exists('model_and_encoders.pkl'):
        os.remove('model_and_encoders.pkl')
        print("Model and encoders cleared.")
    else:
        print("No existing model and encoders found.")

# Function to print final results
def print_final_results(stats_file, classifier_results):
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as file:
            stats = json.load(file)
        print(f"Final Results - Total Predictions: {stats['total']}, Correct Predictions: {stats['correct']}, Accuracy: {stats['correct'] / stats['total']:.2f}")
    else:
        print("No statistics available.")
    
    print("\nClassifier Accuracies:")
    for classifier, accuracy in classifier_results.items():
        print(f"{classifier}: {accuracy:.4f}")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Keystroke Dynamics Data Collection and Analysis. The script performs keystroke data collection and analyzes it to predict the user's identity based on typing patterns. The collected data is saved in a CSV file, and the script can analyze the data using multiple classifiers to determine the best performing model. The script also allows real-time testing where a user types some text, and the model predicts the user's identity.")
    parser.add_argument('--mode', choices=['collect', 'analyze', 'both', 'test'], required=True, help="Mode to run the script in: collect, analyze, both, or test")
    parser.add_argument('--filename', default='keystroke_data.csv', help="Filename for keystroke data CSV (default: keystroke_data.csv)")
    parser.add_argument('--stats_file', default='stats.json', help="Filename for statistics JSON (default: stats.json)")
    parser.add_argument('--users', type=int, default=2, help="Number of users for data collection (default: 2)")
    parser.add_argument('--clear_stats', action='store_true', help="Clear the statistics file")
    parser.add_argument('--clear_model', action='store_true', help="Clear the model and encoders")
    args = parser.parse_args()

    if args.clear_stats:
        clear_stats(args.stats_file)
        return

    if args.clear_model:
        clear_model_and_encoders()
        return

    if args.mode in ['collect', 'both']:
        collect_keystroke_data_for_multiple_users(args.users, args.filename)

    if args.mode in ['analyze', 'both']:
        model, user_encoder, encoder, scaler, classifier_results = analyze_data(args.filename)
    elif args.mode == 'test':
        model, user_encoder, encoder, scaler, classifier_results = load_model_and_encoders()
        if model is None:
            return

    if args.mode in ['analyze', 'both', 'test']:
        while True:
            correct = predict_and_validate_user(model, user_encoder, encoder, scaler, args.stats_file)
            continue_feedback = input("Do you want to continue? (yes/no): ").strip().lower()
            if continue_feedback != 'yes':
                break

        print_final_results(args.stats_file, classifier_results)

if __name__ == "__main__":
    main()
