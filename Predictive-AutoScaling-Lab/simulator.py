import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


# ============================================================
#  Prediction Baselines (outside the class)
# ============================================================

def reactive_policy(cpu, threshold=0.6, hysteresis=0.2):
    """Simple reactive scaling rule."""
    if cpu > threshold:
        return +1
    elif cpu < threshold - hysteresis:
        return -1
    return 0


def shifted_load_prediction(cpu_history):
    """Predict next CPU = last CPU."""
    return cpu_history[-1]


def moving_average_prediction(cpu_history, window=3):
    """Predict next CPU = average of last N points."""
    if len(cpu_history) < window:
        return cpu_history[-1]
    return np.mean(cpu_history[-window:])

def arima_predictor(order=(1,0,0)):
    def predict(cpu_history):
        if len(cpu_history) < 10:
            return cpu_history[-1]
        try:
            model = ARIMA(cpu_history, order=order).fit(method='statespace')
            return max(model.forecast()[0], 0)
        except:
            return cpu_history[-1]
    return predict



# ============================================================
#  AutoScaling Simulator
# ============================================================

class AutoScalingSimulator:
    def __init__(self, actual_load, config):
        self.actual_load = actual_load
        self.capacity = config["capacity_per_instance"]
        self.cost_per_instance = config["cost_per_instance"]
        self.cooldown = config["cooldown"]
        self.instances = config["initial_instances"]
        self.cooldown_timer = 0

        # Logging
        self.history = {
            "instances": [],
            "actual_load": [],
            "predicted_load": [],
            "sla_violation": [],
            "cost": [],
            "scaled_up": [],
            "scaled_down": []
        }

    def apply_action(self, action):
        """Apply scaling action (+1, -1, or 0)."""
        scaled_up = 0
        scaled_down = 0

        if action == +1:
            self.instances += 1
            scaled_up = 1
            self.cooldown_timer = self.cooldown

        elif action == -1 and self.instances > 1:
            self.instances -= 1
            scaled_down = 1
            self.cooldown_timer = self.cooldown

        return scaled_up, scaled_down

    def step(self, t, predicted):
        actual = self.actual_load[t]

        # Cooldown logic
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            action = 0
        else:
            action = reactive_policy(predicted)

        scaled_up, scaled_down = self.apply_action(action)

        # SLA violation
        sla_violation = int(actual > self.instances * self.capacity)

        # Cost
        cost = self.instances * self.cost_per_instance

        # Log
        self.history["instances"].append(self.instances)
        self.history["actual_load"].append(actual)
        self.history["predicted_load"].append(predicted)
        self.history["sla_violation"].append(sla_violation)
        self.history["cost"].append(cost)
        self.history["scaled_up"].append(scaled_up)
        self.history["scaled_down"].append(scaled_down)

    def run(self, predictor_fn):
        """Run simulation using a prediction function."""
        cpu_history = []

        for t in range(len(self.actual_load)):
            cpu_history.append(self.actual_load[t])

            predicted = predictor_fn(cpu_history)
            self.step(t, predicted)

        return pd.DataFrame(self.history)


# ============================================================
#  Experiment Runner
# ============================================================

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("invocations_per_function_md.anon.d01.csv")
    time_cols = [str(i) for i in range(1, 1441)]
    actual_load = df.iloc[0][time_cols].values.astype(float)

## Experiment with different load patterns


# 1) High load scenario for testing
# actual_load = actual_load * 3

# 2) Bursty traffic
# actual_load = actual_load + np.random.randint(0, 200, size=len(actual_load))

# 3) Sudden spikes
# spikes = np.zeros_like(actual_load)
# spikes[np.random.choice(len(actual_load), 10)] = 500
# actual_load = actual_load + spikes






# For now, use a naive predictor (shifted actuals)
predicted_load = np.roll(actual_load, 1)
predicted_load[0] = actual_load[0]

# Config file (this is your experiment seed)
config = {
    "capacity_per_instance": 100,
    "cost_per_instance": 1.0,
    "cooldown": 5,
    "initial_instances": 1
}

# -------------------------
# Run baselines
# -------------------------

print("Running reactive baseline...")
sim_reactive = AutoScalingSimulator(actual_load, config)
results_reactive = sim_reactive.run(lambda hist: hist[-1])  # reactive uses actual CPU
results_reactive.to_csv("reactive_results.csv", index=False)

print("Running shifted-load baseline...")
sim_shifted = AutoScalingSimulator(actual_load, config)
results_shifted = sim_shifted.run(shifted_load_prediction)
results_shifted.to_csv("shifted_results.csv", index=False)

print("Running moving-average baseline...")
sim_ma = AutoScalingSimulator(actual_load, config)
results_ma = sim_ma.run(lambda hist: moving_average_prediction(hist, window=3))
results_ma.to_csv("moving_avg_results.csv", index=False)

print("Running ARIMA baseline...")
sim_arima = AutoScalingSimulator(actual_load, config)
arima_fn = arima_predictor(order=(2,1,2))
results_arima = sim_arima.run(arima_fn)
results_arima.to_csv("arima_results.csv", index=False)


print("Done.")
