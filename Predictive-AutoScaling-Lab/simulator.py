import pandas as pd
import numpy as np

class AutoScalingSimulator:
    def __init__(self, actual_load, predicted_load, config):
        self.actual_load = actual_load
        self.predicted_load = predicted_load
        self.capacity = config["capacity_per_instance"]
        self.cost = config["cost_per_instance"]
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

    def step(self, t):
        predicted = self.predicted_load[t]
        actual = self.actual_load[t]

        scaled_up = 0
        scaled_down = 0

        # Cooldown logic
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        else:
            # Baseline scaling rule
            if predicted > self.instances * self.capacity:
                self.instances += 1
                scaled_up = 1
                self.cooldown_timer = self.cooldown

            elif predicted < (self.instances - 1) * self.capacity:
                self.instances -= 1
                scaled_down = 1
                self.cooldown_timer = self.cooldown

        # SLA violation
        sla_violation = int(actual > self.instances * self.capacity)

        # Cost
        cost = self.instances * self.cost

        # Log everything
        self.history["instances"].append(self.instances)
        self.history["actual_load"].append(actual)
        self.history["predicted_load"].append(predicted)
        self.history["sla_violation"].append(sla_violation)
        self.history["cost"].append(cost)
        self.history["scaled_up"].append(scaled_up)
        self.history["scaled_down"].append(scaled_down)

    def run(self):
        for t in range(len(self.actual_load)):
            self.step(t)
        return pd.DataFrame(self.history)


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("invocations_per_function_md.anon.d01.csv")

    time_cols = [str(i) for i in range(1, 1441)]

    # Pick one app
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
        "capacity_per_instance": 100,   # adjust based on your dataset
        "cost_per_instance": 1.0,
        "cooldown": 5,                  # minutes
        "initial_instances": 1
    }

    sim = AutoScalingSimulator(actual_load, predicted_load, config)
    results = sim.run()

    print(results.head())
    results.to_csv("simulator_output.csv", index=False)
