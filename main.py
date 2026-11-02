# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================
import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.classes import Batch  # This allows you to import individual Classes from classes.py
from src.constants import *  # This imports all constants from constants.py

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(0)

n_simulations = 100

df_ic = pd.read_csv(os.path.join("datasets", "initial_conditions.csv"))
target_batches = list(range(5, 151, 5))
df_30 = df_ic[df_ic["Batch ID"].isin(target_batches)].copy()
df_30["TCD [cells/mL]"] = df_30["VCD [cells/mL]"]
df_30[["Lactate [mM]", "Ammonia [mM]", "mAb [mg/L]"]] = 0
df_30["Culture_Cost"] = (10*df_30["Glucose [mM]"]+20*df_30["Glutamine [mM]"]+0.0005*df_30["VCD [cells/mL]"])

p_max = 5
cv_splits = 3
poly_degree_max = 3
knn_k_max = 5

model_types = ["poly", "knn"]
param_grid_poly = dict(regressor__polynomialfeatures__degree=[i for i in range(1, poly_degree_max + 1)])
param_grid_knn = dict(regressor__kneighborsregressor__n_neighbors=[i for i in range(1, knn_k_max + 1)],
                      regressor__kneighborsregressor__weights=["uniform", "distance"])

all_batches = {}
for i in range(5, 151, 5):
    filename = f"batch_{i:04d}.csv"
    all_batches[f"batch_{i}"] = pd.read_csv(os.path.join("datasets", filename))

df_list = []
for batch_key, batch_df in all_batches.items():
    temp_df = batch_df.copy()
    batch_num = int(batch_key.split("_")[1])
    temp_df["Batch ID"] = batch_num
    df_list.append(temp_df)

master_df = pd.concat(df_list, ignore_index=True)
features = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]", "Lactate [mM]", "Ammonia [mM]", "mAb [mg/L]"]
idx_mab = features.index("mAb [mg/L]")

global_scaler = StandardScaler()

master_df_scaled = master_df.copy()
master_df_scaled[features] = global_scaler.fit_transform(master_df[features])

model_leaderboard = []

for model_type in model_types:
    for p in range(1, p_max + 1):

        x_lag_list, x_cur_list, groups_list = [], [], []

        for b_id in target_batches:
            batch_data = master_df_scaled[master_df_scaled["Batch ID"] == b_id].sort_values(by="Time [h]")

            x_cur_batch = batch_data[features].iloc[p:].values

            lagged_frames = []
            for lag in range(1, p + 1):
                lag_shifted = batch_data[features].shift(lag)
                lagged_frames.append(lag_shifted)

            x_lag_batch = pd.concat(lagged_frames, axis=1).iloc[p:].values

            x_lag_list.append(x_lag_batch)
            x_cur_list.append(x_cur_batch)
            groups_list.append(np.full(len(x_cur_batch), b_id))

        X = np.vstack(x_lag_list)
        y = np.vstack(x_cur_list)
        groups = np.concatenate(groups_list)

        if model_type == "poly":
            model = make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
            param_grid = dict(polynomialfeatures__degree=[i for i in range(1, poly_degree_max + 1)])
        elif model_type == "knn":
            model = KNeighborsRegressor()
            param_grid = dict(n_neighbors=[i for i in range(1, knn_k_max + 1)], weights=["uniform", "distance"])

        gkf = GroupKFold(n_splits=cv_splits)
        model_gscv = GridSearchCV(model, param_grid=param_grid, cv=gkf, scoring="neg_root_mean_squared_error",
                                  n_jobs=-1)
        model_gscv.fit(X, y, groups=groups)

        best_idx = model_gscv.best_index_
        best_hyperparams = model_gscv.best_params_

        model_leaderboard.append({
            "model_type": model_type,
            "p": p,
            "best_params": best_hyperparams,
            "best_rmse": -model_gscv.best_score_,
            "rmse_std": model_gscv.cv_results_["std_test_score"][best_idx],
            "best_estimator": model_gscv.best_estimator_
        })

df_ranks = pd.DataFrame(model_leaderboard).sort_values(by="best_rmse").reset_index(drop=True)
winning_estimator = df_ranks.loc[0, "best_estimator"]
winning_p = int(df_ranks.loc[0, "p"])
winning_params = df_ranks.loc[0, "best_params"]

print("\n--- MODEL LEADERBOARD (STANDARDIZED RMSE) ---")
print(df_ranks[["model_type", "p", "best_params", "best_rmse", "rmse_std"]].to_string(index=False))
if df_ranks.loc[0, "model_type"] == "poly":
    winning_degree = winning_params.get("polynomialfeatures__degree", 1)
    print(f"\n Selected Winner: {df_ranks.loc[0, 'model_type']} (Degree {winning_degree}) with p={winning_p}")
else:
    print(f"\n Selected Winner: {df_ranks.loc[0, 'model_type']} with p={winning_p}")

DOWNTIME_DAYS = 1.0
HOURS_PER_DAY = 24.0
TRAINED_LAG_STEP = 8

max_hours_to_test = int(np.max(master_df["Time [h]"]))
optimization_results = []

for _, batch_row in df_30.iterrows():
    b_id = int(batch_row["Batch ID"])
    cost = batch_row["Culture_Cost"]

    batch_history_scaled = master_df_scaled[master_df_scaled["Batch ID"] == b_id].sort_values(by="Time [h]")

    best_stp = -float('inf')
    best_harvest_hour = 0
    best_mab_yield = 0.0

    measured_hours = batch_history_scaled["Time [h]"].values
    measured_states = batch_history_scaled[features].values

    max_batch_hour = int(np.max(measured_hours))
    all_hours_grid = np.arange(0, max_batch_hour + 1)

    interpolated_states = np.zeros((len(all_hours_grid), len(features)))
    for col_idx in range(len(features)):
        interpolated_states[:, col_idx] = np.interp(
            all_hours_grid, measured_hours, measured_states[:, col_idx]
        )

    continuous_history = {h: interpolated_states[h] for h in all_hours_grid}

    start_hour = winning_p * TRAINED_LAG_STEP

    for current_hour in range(start_hour + 1, max_hours_to_test + 1):
        lagged_rows = []
        for lag_idx in range(1, winning_p + 1):
            target_lag_hour = current_hour - (lag_idx * TRAINED_LAG_STEP)

            lagged_rows.append(continuous_history[target_lag_hour])

        input_features = np.array(lagged_rows).flatten().reshape(1, -1)
        next_state_scaled = winning_estimator.predict(input_features).flatten()

        continuous_history[current_hour] = next_state_scaled

        next_state_real = global_scaler.inverse_transform(next_state_scaled.reshape(1, -1))
        predicted_mab = next_state_real[0, idx_mab]

        if predicted_mab < 0:
            predicted_mab = 0.0

        harvest_time_days = current_hour / HOURS_PER_DAY
        cell_culture_value = 1000 * predicted_mab
        profit = cell_culture_value - cost
        space_time_profit = profit / (harvest_time_days + DOWNTIME_DAYS)

        if space_time_profit > best_stp:
            best_stp = space_time_profit
            best_harvest_hour = current_hour
            best_mab_yield = predicted_mab

    optimization_results.append({
        "Batch ID": b_id,
        "Optimal Harvest Hour": best_harvest_hour,
        "Optimal Harvest Day": round(best_harvest_hour / HOURS_PER_DAY, 2),
        "Predicted mAb [mg/L]": round(best_mab_yield, 2),
        "Max Space-Time Profit [$/(L·d)]": round(best_stp, 2)
    })

df_plan = pd.DataFrame(optimization_results)
print("\n--- OPTIMAL HOURLY BATCH HARVESTING SCHEDULE ---")
print(df_plan.to_string(index=False))

min_glucose, max_glucose = df_30["Glucose [mM]"].min(), df_30["Glucose [mM]"].max()
min_glutamine, max_glutamine = df_30["Glutamine [mM]"].min(), df_30["Glutamine [mM]"].max()
min_vcd, max_vcd = df_30["VCD [cells/mL]"].min(), df_30["VCD [cells/mL]"].max()

test_glucose = np.linspace(min_glucose, max_glucose, num=10)
test_glutamine = np.linspace(min_glutamine, max_glutamine, num=10)
test_vcd = np.linspace(min_vcd, max_vcd, num=10)

virtual_results = []
MAX_SIMULATION_HOURS = 480

print("\nRunning continuous virtual screening across 1,000 custom recipes...")

for glc in test_glucose:
    for gln in test_glutamine:
        for vcd in test_vcd:

            hypothetical_cost = (10 * glc) + (20 * gln) + (0.0005 * vcd)

            day_0_real = pd.DataFrame([[glc, gln, vcd, 0.0, 0.0, 0.0]], columns=features)
            day_0_scaled = global_scaler.transform(day_0_real).flatten()

            continuous_history = {0: day_0_scaled}

            best_virtual_stp = -float('inf')
            best_hour = 0
            best_yield = 0.0

            for current_hour in range(1, MAX_SIMULATION_HOURS + 1):

                lagged_rows = []
                for lag_idx in range(1, winning_p + 1):
                    target_lag_hour = current_hour - (lag_idx * TRAINED_LAG_STEP)

                    if target_lag_hour in continuous_history:
                        lagged_rows.append(continuous_history[target_lag_hour])
                    else:
                        lagged_rows.append(continuous_history[0])

                input_features = np.array(lagged_rows).flatten().reshape(1, -1)

                next_state_scaled = winning_estimator.predict(input_features).flatten()
                continuous_history[current_hour] = next_state_scaled

                next_state_real = global_scaler.inverse_transform(next_state_scaled.reshape(1, -1))
                predicted_mab = next_state_real[0, idx_mab]

                if predicted_mab < 0:
                    predicted_mab = 0.0

                harvest_time_days = current_hour / HOURS_PER_DAY
                profit = (1000 * predicted_mab) - hypothetical_cost
                space_time_profit = profit / (harvest_time_days + DOWNTIME_DAYS)

                if space_time_profit > best_virtual_stp:
                    best_virtual_stp = space_time_profit
                    best_hour = current_hour
                    best_yield = predicted_mab

            virtual_results.append({
                "Glucose": glc, "Glutamine": gln, "VCD": vcd, "Cost": hypothetical_cost,
                "Best Hour": best_hour, "Best Day": round(best_hour / 24, 2),
                "Predicted mAb": best_yield, "Max Virtual STP": best_virtual_stp
            })

df_virtual_screening = pd.DataFrame(virtual_results)
df_theor_opt = df_virtual_screening.loc[df_virtual_screening["Max Virtual STP"].idxmax()]

print("\n" + "=" * 60)
print("             THEORETICAL OPTIMUM RECIPE FOUND                 ")
print("=" * 60)
print(" INITIAL RECIPE TARGETS:")
print(f"  • Target Glucose   : {df_theor_opt['Glucose']:.2f} mM")
print(f"  • Target Glutamine : {df_theor_opt['Glutamine']:.2f} mM")
print(f"  • Target VCD       : {df_theor_opt['VCD']:.0f} cells/mL")
print(f"  • Expected Cost    : {df_theor_opt['Cost']:.2f} $/L")
print("-" * 60)
print(f" Peak Virtual Space-Time Profit : {df_theor_opt['Max Virtual STP']:.2f} $/(L·d)")
print(f" Recommended Harvest Schedule    : Day {df_theor_opt['Best Day']} (Hour {df_theor_opt['Best Hour']})")
print(f" Predicted Target mAb Production: {df_theor_opt['Predicted mAb']:.2f} mg/L")
print("=" * 60)

ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batch = Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal)
batch.display_info()
