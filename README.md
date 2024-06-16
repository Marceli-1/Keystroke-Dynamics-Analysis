# Keystroke-Dynamics-Analysis
Keystroke Dynamics Data Collection and Biometric Identification using Machine Learning Models

This script allows you to collect and analyze keystroke dynamics data for multiple users.
It supports four modes of operation: 'collect', 'analyze', 'both', and 'test'.

### Modes:
- **collect**: Collect keystroke data for multiple users and save it to a CSV file.
- **analyze**: Analyze the collected keystroke data, train a machine learning model, and predict user identity.
- **both**: Perform both data collection and analysis.
- **test**: Test the user identity prediction using an existing model.

Usage:
- **Collect data**: 
  ```python3 biometric-keystrokes.py --mode collect --filename keystroke_data.csv --users 2```
- **Analyze data**: 
  ```python3 biometric-keystrokes.py --mode analyze --filename keystroke_data.csv```
- **Collect and analyze data**: 
  ```python3 biometric-keystrokes.py --mode both --filename keystroke_data.csv --users 2```
- **Test user identity**: 
  ```python3 biometric-keystrokes.py --mode test --stats_file stats.json --model_file model_and_encoders.pkl```

Files:
- **keystroke_data.csv**: CSV file to store collected keystroke data.
- **model_and_encoders.pkl**: File to save the trained model and encoders.
- **stats.json**: JSON file to store prediction statistics.

Options:
- **--mode**: Mode to run the script in: collect, analyze, both, or test.
- **--filename**: Filename for keystroke data CSV.
- **--stats_file**: Filename for statistics JSON.
- **--users**: Number of users for data collection.

Additional Options:
- **--clear_stats**: Clear the statistics file.
- **--clear_model**: Clear the model and encoders file.

For more information, please refer to the script documentation or run the script with the --help option.

