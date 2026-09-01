"""AQI Forecasting demo using a reproducible synthetic dataset."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def build_dataset(n_samples=1000):
    """Create reproducible sample environmental measurements."""
    rng = np.random.default_rng(RANDOM_STATE)

    pm25 = rng.uniform(10, 250, n_samples)
    pm10 = rng.uniform(20, 350, n_samples)
    no2 = rng.uniform(5, 120, n_samples)
    co = rng.uniform(0.2, 5.0, n_samples)
    temperature = rng.uniform(12, 42, n_samples)
    humidity = rng.uniform(25, 90, n_samples)

    # Educational synthetic relationship; not an official AQI formula.
    aqi = (
        0.55 * pm25
        + 0.22 * pm10
        + 0.65 * no2
        + 8 * co
        + 0.15 * temperature
        - 0.08 * humidity
        + rng.normal(0, 8, n_samples)
    )
    aqi = np.clip(aqi, 0, None)

    return pd.DataFrame(
        {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "CO": co,
            "Temperature": temperature,
            "Humidity": humidity,
            "AQI": aqi,
        }
    )


def main():
    data = build_dataset()
    features = ["PM2.5", "PM10", "NO2", "CO", "Temperature", "Humidity"]

    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data["AQI"], test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.6)
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(limits, limits, linestyle="--")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI")
    plt.tight_layout()
    plt.savefig("aqi_actual_vs_predicted.png", dpi=150)
    print("Saved chart: aqi_actual_vs_predicted.png")


if __name__ == "__main__":
    main()
