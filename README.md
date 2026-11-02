# Bioprocess NVAR Optimizer
This code uses machine learning to predict how cells grow and make antibodies in a bioreactor over time.
It solves a big problem for biochemical engineers: knowing exactly when to harvest a batch to make the most money, and finding the perfect starting recipe for the nutrients.

## Highlights
* Built a Nonlinear Vector Autoregressive (NVAR) time-series pipeline to model cell culture kinetics from experimental datasets.
* Optimized bioreactor harvest schedules and initial nutrient concentrations to maximize space-time profit using automated simulations.
* Developed a cross-validation grid search strategy to compare polynomial and K-Nearest Neighbors models.
* Standardized multi-unit target matrices across mismatched engineering scales using advanced preprocessing techniques to ensure uniform feature weighting.
* Evaluated model stability over time by mapping mean and standard deviation metrics.

## Environment
* Python 3.14.7
* numpy 2.5.2
* pandas 3.0.5
* scikit-learn 1.9.0

## Code Design
Executing `main.py` launches an automated, data-driven optimization pipeline.
The software combines multi-variable time-series forecasting with economic objective modeling.
Rather than relying on a single-point prediction, the code sequentially transforms raw bioreactor batch tracking data into an optimized pipeline.
The following sections outline the major stages of this workflow.
### Data Aggregation and Collection
This stage locates and extracts the experimental dataset from the storage directories. It loads the global project variables (`initial_conditions.csv`) containing the seeding metrics for the target batches.
It then loops through the datasets folder to programmatically import individual batch data spreadsheets (`batch_0005.csv` to `batch_0150.csv`). The separate runs are tagged with their specific Batch ID and concatenated into a unified dataframe (`master_df`).
The experimental bioprocess data is separated through many different files. Each batch represents an isolated fermentation run recorded in a separate spreadsheet. Machine learning models cannot train on directory structures or individual files. This stage consolidates all the data into one dataframe, providing a unique location for all the data used to train the model.
This stage establishes the data foundation for the application. It maps the timelines and identifies which rows belong to which batch. By binding the data together with a categorical group tag (Batch ID), it enables the cross-validation system to partition data safely without destroying the chronological relationships of the individual runs.
### Preprocessing and Multivariable Scaling
This stage standardizes the numerical features. The bioreactor variables use drastically different units—Viable Cell Density (VCD) spans millions of cells, whereas Monoclonal Antibody (mAb) yields exist as small decimal numbers. This block fits a global `StandardScaler` to the dataset, shifting and scaling every feature so that each tracking column has a mean of zero and a variance of one.
Non-standardized scales distort machine learning algorithms. If left unscaled, a model calculating error vectors will be blinded by the massive numbers in the VCD column. A tiny 1% tracking error in cell count would numerically overwhelm a massive 50% calculation error in antibody yield, rendering the model unable to accurately forecast product accumulation.
Scaling variables up front standardizes the mathematical weight of every process component. It ensures that the optimization engine will treat all the variables in cellular growth kinetics with the same importance. This results in stable model training and allows the cross-validation scoring loop to return clean, unskewed error metrics.
### Autoregressive Feature Engineering
This stage takes the time-course rows and turns them into a matrix. To build a Vector Autoregressive (VAR) structure of lag order *p*, the code steps through each batch independently. It isolates the data at time $t$ while sliding the other states down to $t-1$, $t-2$, and so on to generate historical input features representing the time series data.
Bioreactor cultures are dynamic biological systems; the future state of a cell culture depends on its past history. Shifting the data frames transforms a time-series log into explicit state-transition pairs. Slicing the lags strictly within batch boundaries prevents the last hour of one run from bleeding into the first hour of the next.
This step maps the input-output architecture for the predictive models. It directly determines the feature dimensions ($p \times 6$ variables) entering the training grid. This structure enables the model to learn the fundamental biological kinetics rather than simply memorizing the data and making unrealistic predictions if the conditions are slightly different.
### Grid Search and Model Evaluation
This step executes a hyperparameter optimization sweep across different model types (Polynomial Linear Regression and K-Nearest Neighbors) and lag horizons ($p$) ranging from 1 to 5 lagged values. It configures a cross-validator to split the training pool by Batch ID. A `GridSearchCV` engine then tests every configuration of polynomial model from degree 1 to 3 and k-Nearest Neighbors from $k=1$ to $k=5$ across the data splits, captures the root mean squared error and the standard deviations, and prints a sorted performance leaderboard.
No single algorithm fits every biological profile perfectly. High-degree polynomials can overfit and oscillate, while misconfigured KNN models can lose subtle kinetic trends. Evaluating models using Group Cross-Validation ensures that the selected model is chosen based on its ability to learn biological trends and generalize to  unseen batches, rather than its capacity to overfit familiar training data.
This block serves as the automated decision maker of the pipeline. It scores and ranks every candidate model, filters out models prone to variance or noise, and isolates the highest-performing estimator along with its optimal lag horizon ($p$). This winning architecture is then locked in to drive the downstream optimization simulations.
### Profit Optimization on Training Batches
This stage calculates the optimal harvest schedule for the empirical batches. It reconstructs the raw discrete timelines into a continuous, 1-hour resolution grid using linear interpolation. It then uses the winning model to forecast the bioreactor state hour-by-hour. At each simulated hour, the script reconverts the data to real-world values, calculates dollar yields, subtracts the media seeding costs, applies the 1-day vessel downtime penalty, and identifies the exact hour that maximizes space-time profit ($/L\cdot d$).
In industrial bioprocessing, maximizing raw antibody output does not guarantee maximum profit. Running a bioreactor too long incurs a steep time penalty that degrades the overall economic efficiency of the facility. Conversely, harvesting too early wastes costly starting nutrients. This stage models these conflicting trade-offs over time to identify the exact point where production returns begin to diminish relative to the time invested. The program then generates a table with the results for all 30 experimental batches.
This stage translates abstract machine learning predictions into operational decisions. By combining the model's kinetic forecasts with the financial parameters of the process, it generates an optimized harvesting schedule for every batch, detailing the precise hour to stop the run to obtain the best possible economic value for each set of initial conditions.
### Recipe Optimization via Simulation
The final stage of the workflow performs a virtual simulation across thousands of hypothetical starting formulations. It generates a multidimensional grid bounded tightly by the minimum and maximum nutrient levels observed in the historical batches. The system seeds the model with these new synthetic initial conditions, runs a full multi-day simulation for each recipe combination, and evaluates their maximum achievable space-time profits to output a single global "optimal recipe."
Relying strictly on historical data limits optimization to combinations that have already been tried. The advantage of a proper machine learning model is its ability to predict outputs from new data. This step allows engineers to safely test completely new, unrun ratios of initial glucose, glutamine, and seeding cell densities without wasting raw materials, incurring bioreactor downtime, or risking batch failures.
This stage completes the optimization loop by shifting the focus from scheduling to process design. It delivers a concrete, data-backed recipe for the ideal starting conditions. This gives engineers a precise target for configuring future bioreactor runs to maximize the economic efficiency of the monoclonal antibody production process.

## Analysis
### Justification of Batch Selection and Initial Conditions
To train a robust machine learning model within the constraints of a 30-batch budget, the choice of initial conditions must prioritize design space diversity over randomness. The available candidate runs from the dataset were filtered to extract a grid of 30 distinct historical trajectories (spaced uniformly by batch IDs: 5, 10, 15, ..., 150). This specific selection strategy spans a wide range of diverse initial conditions and avoids using any of the duplicate runs.
Testing a broad range of starting states is essential because cell growth kinetics are highly dependent on these initial conditions. If the 30 batches were selected strictly from a narrow cluster (e.g., only high-glucose runs), the model would remain completely blind to the effects of nutrient starvation or metabolic inhibition. Additionally, the designed model is only effective in the range of conditions it was allowed to train on making it essential to test various starting conditions.
By sampling the full spectrum of experimental bounds, the dataset captures vital edge cases where nutrient depletion caps cell reproduction or where over-feeding drives toxic accumulation of byproduct metabolites like Lactate and Ammonia. This systematic structural coverage provides the exact training variety required for the model to successfully generalize and interpolate transitions for completely new, unrun recipes.
### Model Architecture and Hyperparameter Justification
The selected model architecture is a linear Vector Autoregressive model using 5 lagged values implemented via a custom Scikit-Learn pipeline consisting of `PolynomialFeatures(degree=1, include_bias=False)` bound to a multi-output `LinearRegression` engine. The entire process matrix is mapped through a unified `StandardScaler` up front. This configuration means that at any given hour $t$, the next hourly state vector containing all 6 process metrics is predicted using a linear combination of those same 6 variables tracked across the past 5 consecutive 8-hour milestones (representing a rolling 40-hour window).
The hyperparameter selection was determined objectively by executing an automated grid search evaluated via 3-fold `GroupKFold` cross-validation. Grouping data by Batch ID ensures that the validation folds assess out-of-sample prediction strength on entirely unvisited bioreactor trajectories. While the pipeline evaluated polynomial configurations up to degree 3, the cross-validation scoring loop identified that a degree of 1 achieved the lowest global standardized RMSE (0.1341) with an extremely tight standard deviation ($\pm$ 0.0029). 
Opting for a degree-1 linear structure within the NVAR framework is scientifically justified because higher-order polynomials introduce high variance. Squaring or cubing features over a rolling 40-hour window can cause extreme cases where massive overfitting and wild oscillations are present when the model is asked to project far into the future. A linear mapping on top of multi-lag histories acts as a stable equation that predicts accurately while also giving a logical meaning to all the feature coefficients. It is the best model to see clearly the effect of each variable on the production of monoclonal antibodies.
### Comparative Alternative Analysis
To verify the selection of the NVAR Linear pipeline, it was benchmarked directly against higher order polynomial models and a multi-output K-Nearest Neighbors Regressor (`KNeighborsRegressor`) operating across identical lag windows. The KNN pipeline evaluated neighbor spaces ($k$) from 1 to 5 and compared uniform weights against inverse distance-based weight assignments. The cross-validation leaderboard clearly shows that the KNN configurations underperformed relative to the polynomial NVAR equations across every lag metric:
| Rank | Model Type | Lag Order ($p$) | Optimal Parameters | Cross-Validation RMSE | Validation SD ($\pm$) |
| :---: | :--- | :---: | :--- | :---: | :---: |
| 1 | Polynomial | 5 | `{'degree': 1}` | 0.134175 | 0.002948 |
| 2 | Polynomial | 4 | `{'degree': 1}` | 0.137530 | 0.003870 |
| 3 | Polynomial | 3 | `{'degree': 1}` | 0.142812 | 0.004317 |
| 4 | Polynomial | 2 | `{'degree': 1}` | 0.152344 | 0.005414 |
| 5 | Polynomial | 1 | `{'degree': 1}` | 0.170069 | 0.007474 |
| 6 | KNN | 2 | `{'n_neighbors': 5, 'weights': 'distance'}` | 0.274649 | 0.068360 |
| 7 | KNN | 3 | `{'n_neighbors': 5, 'weights': 'distance'}` | 0.274866 | 0.069601 |
| 8 | KNN | 4 | `{'n_neighbors': 5, 'weights': 'distance'}` | 0.275900 | 0.069574 |
| 9 | KNN | 5 | `{'n_neighbors': 5, 'weights': 'distance'}` | 0.277662 | 0.070408 |
| 10 | KNN | 1 | `{'n_neighbors': 5, 'weights': 'distance'}` | 0.280992 | 0.069849 |

The KNN model's average error is more than double that of the linear system (0.2746 vs 0.1341), and its validation variance is significantly wider ($\pm$ 0.068 vs $\pm$ 0.003). KNN fails in this specific time-course workflow because it is a non-parametric instance-based learner. It relies entirely on interpolating localized distance points within the historical training rows.
Because the 30-batch dataset contains data that is very different when cut hour-by-hour, the KNN model frequently snaps to distant, mismatched batch trajectories when evaluating possible outputs. This breaks the smooth progression of the biological curves and causes jagged transitions during rolling forecasts.
A more complete dataset might have rendered better results for the kNN models, but the resources available did not permit that.
As for the other polynomial models, the table shows the best performing models with the most optimized hyperparameters for both polynomial and kNN models.
Since all models shown are either degree 1 polynomial or 5 neighbors distance-based weights models, it shows that these are the superior choice in their own categories.
### Advantages and Limitations
The primary advantage of this NVAR linear model is its exceptional numerical stability and computational efficiency. Because it fits standard linear slope coefficients across standardized historical lag steps, it evaluates thousands of virtual screening combinations in seconds. It captures the true underlying biological feedback loops—such as how the accumulation of toxic Lactate and Ammonia slows down the Viable Cell Density replication rate—without requiring complex kinetic equations.
Furthermore, by evaluating tracking variables on a standardized scale, it naturally resolves unit-scale imbalances, giving cell counts and antibody yields equal optimization importance.
Another benefit of having a simpler linear model is its resistance to overfitting training data and generating ridiculous output predictions based on exponentially scaling noise.
The main limitation of this model is its vulnerability to long-horizon error propagation. Because it is optimized exclusively on its one-step-ahead prediction error (predicting hour $t$ from $t-1$), it does not natively account for how errors compound when its own outputs are fed recursively back into its inputs.
Over a 20-day (480-hour) rolling simulation, minor biases can gradually compound, causing the tracking parameters to diverge from realistic biological boundaries. It remains an interpolative tool; it cannot safely extrapolate trends for radical initial conditions that sit completely outside the minimum or maximum parameter boundaries established by the 30 training batches.
### Influence of Initial Conditions on Economic Metrics
The starting configuration of a bioreactor dictates the trajectory of its space-time yield and final profitability. Seeding choices represent a direct financial trade-off between upfront investment and metabolic longevity:
* **Initial Glucose and Glutamine:** These act as the primary energy and carbon sources driving cellular metabolism. High initial levels provide the fuel necessary to sustain dense cell populations over extended cultivation windows, resulting in a large raw antibody yield. However, over-concentrating these sugars heavily penalizes the system in two ways: it drives up the upfront media cost linearly, and it accelerates the cellular production of toxic byproducts like Lactate and Ammonia, which can drastically inhibit production.
* **Initial Viable Cell Density (VCD):** Higher starting cell densities reduce the lag phase of the bioreactor, pushing the culture into exponential antibody production much faster. This drastically shortens the required harvest time, which minimizes the time-denominator of the space-time yield equation. However, a high VCD requires an immense amount of starting nutrients just to survive, leading to rapid starvation if not perfectly balanced with glucose. It also scales the initial formulation cost up by 0.0005$ per cell/mL.
Maximum space-time profit is achieved when these trade-offs are perfectly balanced. The optimal recipe must choose starting levels high enough to maximize antibody expression, but low enough to control media costs and delay byproduct toxicity, ensuring peak production is achieved before the 1-day vessel downtime penalty drives down the daily return rate.
### Global Bioprocess Optimization Targets
By executing a virtual simulation across thousands of recipe combinations, the NVAR optimization engine isolated the absolute peak economic operating conditions.
The definitive process recommendations and estimated outcomes are structured below:
* **Recommended Operational Initial Recipe:**
  * Initial Glucose Concentration: 5.00 mM
  * Initial Glutamine Concentration: 1.00 mM
  * Initial Viable Cell Density (VCD): 200,000 cells/mL
* **Optimal Scheduling Control:**
  * Recommended Harvest Time: 2.71 days (65.0 hours)
* **Estimated Project Economic Balance:**
  * Fixed Cell Culture Seeding Cost: 170.00 $/L
  * Estimated Cell Culture Value at Harvest: 460 $/L with 95% prediction intervals
  * Estimated Max Space-Time Profit: 78.79 \$ /(L $\cdot$ d) with 95% prediction intervals
  * Estimated mAb Space-Time Yield: 0.12 mg/(L $\cdot$ d) with 95% prediction intervals

