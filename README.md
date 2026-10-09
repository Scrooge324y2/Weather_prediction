# Rain Prediction with a Neural Network (PyTorch)

A binary classifier that predicts whether it will rain tomorrow from daily weather observations, built with PyTorch.

## Dataset

The data file is not included in this repository. The script expects a CSV named `weather.csv` in the project root with these columns:

`MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RISK_MM, RainTomorrow`

- 17 input features, ~[N] rows after dropping rows with missing values
- Target: `RainTomorrow` (1 = rain, 0 = no rain)
- `RISK_MM` (the amount of rain the following day) is excluded from the features because it would leak the answer into the model.

## Approach

1. Load the data and drop rows with missing values
2. Split into train/test sets (80/20, `random_state=42`)
3. Standardise features with `StandardScaler`, **fitted on the training set only** to avoid data leakage
4. Train a feed-forward neural network:
   - Architecture: `17 → 10 → 5 → 1` with ReLU activations
   - Loss: `BCEWithLogitsLoss`
   - Optimiser: Adam (lr = 0.001)
   - 100 epochs, batch size 32

## Results

| Metric | Value |
|---|---|
| Train accuracy | [XX.XX]% |
| Test accuracy | [XX.XX]% |

Baseline (always predicting "no rain"): [XX.XX]%.

## Limitations

- The dataset is small (a few hundred rows), so test accuracy is noisy and varies with the random split.
- Classes are imbalanced (more dry days than rainy ones), so accuracy alone can be misleading.
- Rows with missing values are dropped rather than imputed.


## How to run

```bash
git clone https://github.com/Scrooge324y2/Weather_prediction.git
cd Weather_prediction
pip install -r requirements.txt
# add your weather.csv to the project folder
python weather_prediction.py
```

The script prints loss and accuracy every 10 epochs.

