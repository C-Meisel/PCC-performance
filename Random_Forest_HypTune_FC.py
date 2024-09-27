'''
This script is used to tune the hyperparameters
of the fuel cell RFR model
Written with the help of ChatGPT
C-Meisel
'''

' Imports '
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import optuna

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# - Font stuff
from matplotlib import font_manager
SSP_r = '/Library/Fonts/SourceSansPro-Regular.ttf'
SSP_b = '/Library/Fonts/SourceSansPro-Bold.ttf'
ubuntu = '/Library/Fonts/Ubuntu-Regular.ttf'
ubuntu_b = '/Library/Fonts/Ubuntu-Bold.ttf'

font_manager.fontManager.addfont(SSP_r)
font_manager.fontManager.addfont(SSP_b)
font_manager.fontManager.addfont(ubuntu)
font_manager.fontManager.addfont(ubuntu_b)

plt.rcParams['font.family'] = 'Ubuntu' 

hyp_tune_plots = True
trials = 500
optimizer = 'oob' # 'rmse' 'oob'

cv = 5
rs = 76

' - Loading Data - '
data_path = 'Path to Performance_data_pub.xlsx'
sheet_name = 'LC_FC_n9_sel_s'

df_data = pd.read_excel(data_path, sheet_name=sheet_name)

if df_data.columns[-1] == 'PPD_err (W/cm2)':
    err = True
    err_col = df_data.columns[-1]
    ppd_err = df_data[err_col].to_numpy()

    # Drop the last column from the DataFrame
    df = df_data.drop(columns=[err_col])

# - No need to Standardize!
X = df.drop(columns=['Peak power density (W/cm2)'])
y = df['Peak power density (W/cm2)']

' ------- Encoding Catagorical columns (OHE)'
object_cols = X.select_dtypes(include=['object']).columns.tolist() # Get a list of all the object (categorical) columns
for col in object_cols: #  Iterate through the object columns and apply one-hot encoding
    one_hot = pd.get_dummies(X[col], prefix=col, drop_first=False)  # drop_first=True to avoid multicollinearity
    X = pd.concat([X, one_hot], axis=1)
    X.drop(col, axis=1, inplace=True)  # Drop the original categorical column

# Convert boolean values to 0s and 1s
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)

X = X.dropna() # Dropping all N/A values

# Objective function to minimize
def objective(trial):
    global X, y, ppd_err, cv, rs
    X_hyp = X
    y_hyp = y
    ppd_err_hyp = ppd_err
    cv_hyp = cv
    rs_hyp = rs

    n_estimators = trial.suggest_int('n_estimators', 50, 100)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = 1 # trial.suggest_int('min_samples_leaf', 1, 3)  1
    ccp_alpha = 0 #trial.suggest_float('ccp_alpha', 0.001, 0.01)
    rs = rs_hyp # trial.suggest_int('rs', 1, 100) rs_hyp
    cv = cv_hyp # trial.suggest_int('cv',3,7)

    # Initialize the Random Forest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha = ccp_alpha,
        oob_score = True
    )

    # Cross-validation setup
    kf = KFold(n_splits=cv, shuffle=True, random_state=rs)
    oob_scores = []
    y_test_all = []  # To store all actual values
    y_pred_all = []  # To store all predicted values

    # Ensure X and y are numpy arrays
    if isinstance(X_hyp, pd.DataFrame):
        X_hyp = X_hyp.values
    if isinstance(y_hyp, pd.Series):
        y_hyp = y_hyp.values
    if isinstance(ppd_err_hyp, pd.Series):
        ppd_err_hyp = ppd_err_hyp.values

    for train_index, test_index in kf.split(X_hyp):
        X_train, X_test = X_hyp[train_index], X_hyp[test_index]
        y_train, y_test = y_hyp[train_index], y_hyp[test_index]

        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred_fold = model.predict(X_test)

        # Store actual and predicted values
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred_fold)

        # Collect OOB score
        oob_scores.append(model.oob_score_)

    # Compute the average OOB score
    avg_oob_score = sum(oob_scores) / len(oob_scores)
    rmse_cv = np.sqrt(mean_squared_error(y_test_all, y_pred_all))

    if optimizer == 'rmse':
        score = rmse_cv
    elif optimizer == 'oob':
        score = avg_oob_score

    return score

# Create study and optimize
if optimizer == 'rmse':
    direction = 'minimize'
elif optimizer == 'oob':
    direction = 'maximize'

study = optuna.create_study(direction=direction)
study.optimize(objective, n_trials=trials)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")


if hyp_tune_plots == True:
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice
    )

    # Plot optimization history
    fig1 = plot_optimization_history(study)
    fig1.show()

    # Plot parameter importances
    fig3 = plot_param_importances(study)
    fig3.show()

    # Plot slice plot
    fig5 = plot_slice(study)
    fig5.show()

best_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_features=best_params['max_features'],
    min_samples_split=best_params['min_samples_split'],
    # min_samples_leaf=best_params['min_samples_leaf'],
    #ccp_alpha=best_params['ccp_alpha'],
    # random_state=best_params['rs'],
    # cv =best_params['cv'],
    oob_score=True,
)
