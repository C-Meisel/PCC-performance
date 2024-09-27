'''
This script is used to implement a Gaussian Process (GP) model of PCC performance data
The model can analyze CD data gathered from 4411 cells tested at Mines
Written with the help of ChatGPT
C-Meisel
'''

' ----- Imports ----- '
# - Python packages:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtrans
import numpy as np
import matplotlib.gridspec as gridspec
import os
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# - Stat stuff
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.preprocessing import StandardScaler,OneHotEncoder, QuantileTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from scipy.stats import norm

import optuna

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

# - misc functions
def calc_nll(y_true, y_pred, y_std, y_obs_std):
    # Combine model uncertainty and measurement uncertainty
    combined_std = np.sqrt(y_std**2 + y_obs_std**2)
    
    # Calculate NLL
    nll = -np.mean(norm.logpdf(y_true, loc=y_pred, scale=combined_std))
    
    return nll

def is_positive_semi_definite(matrix):
    """Check if a matrix is positive semi-definite."""
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    # Check if all eigenvalues are non-negative

    return np.all(eigenvalues >= -1e-10) 

def get_bar_color(feature_name): # A function to assign colors based on feature name prefixes
    neg_prefixes = ['Negatrode batch', 'Negatrode NiO', 'NiO particle size (μm)', 'NiO (wt%)','BCZYYb (wt%)',
                           'NFL','Negatrode pellet number','Negatrode thickness (mm)','Days (Press to sinter)']
    elyte_prefixes = ['Electrolyte batch','Electrolyte particle size D50 (um)','Electrolyte application',
                      'Electrolyte thickness to grain size ratio','Electrolyte spray solution','Electrolyte spray batch',
                      'Electrolyte treatment', 'Days (Spray to sinter)']
    pos_prefixes = ['PFL','BCFZY batch','Positrode paste','Positrode thickness (μm)','NiO in electrolyte','Positrode sinter furnace',
                    'Positrode paste age (Days)','Positrode sinter temperature (°C)','Positrode sinter batch',
                    'Days (Positrode application to sinter)']
    co_sinter_prefixes = ['Sintering temperature (°C)','Co-sinter furnace','Dried before co-sinter','Two-step sinter',
                          'Co-sinter batch','Absolute humidity at co-sinter (g/m^3)','Sintering neighbor', 'Days (Co-sinter to test)',
                          'Absolute humidity at co-sinter (g/m$^3$)']
    testing_prefixes = ['Test air flow (SCCM)','Test stand','Silver spring','Silver grid paste']
    
    # Check if the feature name starts with any of the prefixes
    # Colors from a colorblind friendly catagorical colorsheme: http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette    
    if any(feature_name.startswith(prefix) for prefix in neg_prefixes):
        return '#009E73'  # Green negatrode
    
    elif any(feature_name.startswith(prefix) for prefix in elyte_prefixes):
        return '#E69F00'  # Orange for electrolyte

    elif any(feature_name.startswith(prefix) for prefix in pos_prefixes):
        return '#56B4E9' # Positrode blue
    
    elif any(feature_name.startswith(prefix) for prefix in co_sinter_prefixes):
        return '#CC79A7' # pink for cosinter
    
    elif any(feature_name.startswith(prefix) for prefix in testing_prefixes):
        return '#F0E442' # Yellow for testing
    
    else:
        return '#333333'  # Default color (Dark Grey)

' -_-_-_-_-_-_-_ Lines to Edit '
file_path = 'Path to Performance_data_pub.xlsx'
sheet_name = 'LC_EC_n64_sel_s'

ls = 1 # This will get automatically tuned by the Kernel
alpha = 0.0028 #
length_scale_bounds = (1e-1, 1e4)
nu = 1.5 # 0.5, 1.5, 2.5
random_state = 35
num_repeat = 10
cv = 5

hyp_tune = False
optuna_trials = 150
z_score = True # If false use quantile transformer

mercer_cond = True

# ---- Visualization
hyp_tune_plots = True
plot_covariance_matrix = False

' - Goodness of Fit'
all_gof = True

' - Feature importance'
print_feature_importance = True
weighted_fi = False
fi_cv = True
plot_feature_importance = True

plot_pdp = False
plot_specific_pdp = None # 'Days (Spray to sinter)'
features_to_plot = 15
top_bottom_features = 10

' - Saving figures - '
save_folder_loc = 'Location of the folder where the graphs'

save_fi = None # os.path.join(save_folder_loc,'EC_FI_GP_figure_name.png')
save_gof = None #os.path.join(save_folder_loc,'EC_GOF_GP_name.png')

' -_-_-_-_-_-_-_ End lines to Edit '

# - Initializing Dataframe
df_data = pd.read_excel(file_path, sheet_name=sheet_name)

if df_data.columns[-1] == 'Current err (A/cm2)':
    err = True
    err_col = df_data.columns[-1]
    ec_err = df_data[err_col].to_numpy()

    # Drop the last column from the DataFrame
    df = df_data.drop(columns=[err_col])

# Separate the features and target variable
X = df.drop(columns=['Current at 1.3V (A/cm2)'])
y = df['Current at 1.3V (A/cm2)']

n_cols = X.shape[1]

print('Number of parameters in the model: ',n_cols)

' ------- Z-score standarization of all numerical features'
if z_score == True:
    # - Initialize the StandardScaler
    scaler = StandardScaler()
else:
    # - Initialize QuantileTransformer
    n_samples = len(y)
    n_quantiles = min(1000, n_samples)  # This ensures n_quantiles is not greater than n_samples
    scaler = QuantileTransformer(n_quantiles=n_quantiles,output_distribution='normal') # ,output_distribution='normal'

# Identify and standardize numerical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

' ------- Encoding Catagorical columns'
object_cols = X.select_dtypes(include=['object']).columns.tolist() # Get a list of all the object (categorical) columns
for col in object_cols: #  Iterate through the object columns and apply one-hot encoding
    one_hot = pd.get_dummies(X[col], prefix=col, drop_first=False)  # drop_first=True to avoid multicollinearity
    X = pd.concat([X, one_hot], axis=1)
    X.drop(col, axis=1, inplace=True)  # Drop the original categorical column

# Convert boolean values to 0s and 1s
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)

X = X.dropna() # Dropping all N/A values
n_col_cat = X.shape[1]

print('Number of encoded parameters in the model: ',len(X.columns))

X.rename(columns={'Absolute humidity at co-sinter (g/m^3)': 'Absolute humidity at co-sinter (g/m$^3$)'},
          errors='raise',  inplace=True) # For making the graphs look better

' ----- Preparing the data to be placed into a model'
# Split the data into training and testing sets
X_train, X_test, y_train, y_test, err_train,err_test = train_test_split(X, y,ec_err, test_size=1/cv, random_state=random_state)
kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

if hyp_tune == True:
    ' ------- Optuna tuning'
    def objective_nll(trial):
        global X, y, ec_err
        X_hyp = X
        y_hyp = y
        ec_err_hyp = ec_err

        # Suggest values for hyperparameters - Uncomment alpha and nu to tune them too
        # alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
        # nu = trial.suggest_categorical('nu', [0.5,1.5,2.5])  # Different values for the smoothness parameter
        random_state = trial.suggest_int('random_state', 0, 100)

        # Define the kernel and model with the suggested hyperparameters
        kernel =  Matern(length_scale=ls, nu=nu,length_scale_bounds=length_scale_bounds)
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=alpha) 

        # Custom cross-validation for NLL
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        residuals = [] # Arrays to store the residuals for each fold
        y_test_all = []  # To store all actual values
        y_pred_all = []  # To store all predicted values
        y_std_all = []

        # Ensure X and y are numpy arrays
        if isinstance(X_hyp, pd.DataFrame):
            X_hyp = X_hyp.values
        if isinstance(y_hyp, pd.Series):
            y_hyp = y_hyp.values
        if isinstance(ec_err_hyp, pd.Series):
            ec_err_hyp = ec_err_hyp.values
            
        for train_index, test_index in kf.split(X_hyp):
            X_train, X_test = X_hyp[train_index], X_hyp[test_index]
            y_train, y_test = y_hyp[train_index], y_hyp[test_index]

            # Fit the model
            gp_model.fit(X_train, y_train)

            # Predict on the test set
            y_pred_fold, y_std_fold = gp_model.predict(X_test, return_std=True)

            # Store actual and predicted values
            y_test_all.extend(y_test)
            y_pred_all.extend(y_pred_fold)
            y_std_all.extend(y_std_fold)

        # Convert lists to numpy arrays for easy plotting
        residuals = np.array(residuals)
        y_test_all = np.array(y_test_all)
        y_pred_all = np.array(y_pred_all)
        y_std_all = np.array(y_std_all)

        nll_cv = calc_nll(y_test_all, y_pred_all, y_std_all, ec_err)
        rmse_cv = np.sqrt(mean_squared_error(y_test_all, y_pred_all))

        return nll_cv
        
    # Create a study object and optimize the objective function for NLL
    study_nll = optuna.create_study(direction='minimize')
    study_nll.optimize(objective_nll, n_trials=optuna_trials)
    best_params = study_nll.best_params
    study = study_nll

    print(f"Best hyperparameters: {best_params}")

    # ---  Re-set hyps: Any hyperparameter being tuned should be active here (un-commented)
    # gpr_alpha = best_params['alpha']
    # nu = best_params['nu']
    random_state = best_params['random_state']

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

# Define the Gaussian Process Regression model with RBF kernel
kernel = Matern(length_scale=ls, nu=nu,length_scale_bounds=length_scale_bounds) # 
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha = alpha) # err_train

# Train the model
X_train, X_test, y_train, y_test, err_train, err_test = train_test_split(X, y,ec_err, test_size=1/cv, random_state=random_state)
kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

gp_model.fit(X_train, y_train)

if mercer_cond == True:
    K = kernel(X, X)
    # Check positive semi-definiteness
    if is_positive_semi_definite(K):
        print("The kernel matrix is positive semi-definite, satisfying Mercer's condition.")
    else:
        print("The kernel matrix is not positive semi-definite, failing Mercer's condition.")

print("Length scale:", gp_model.kernel_.length_scale)

' ========= Feature importance ========= '
if print_feature_importance == True:
    if fi_cv == False:
        perm_importance = permutation_importance(gp_model, X_test, y_test,
                                                n_repeats=num_repeat, random_state=random_state, scoring='neg_root_mean_squared_error')

    else:
        importances = np.zeros(X.shape[1])

        # Iterate through the KFold splits
        # Ensure X and y are numpy arrays
        if isinstance(X, pd.DataFrame):
            X_a = X.values
        if isinstance(y, pd.Series):
            y_a = y.values
        if isinstance(ec_err, pd.Series):
            ec_err = ec_err.values

        for train_index, test_index in kf.split(X_a):
            X_train_cv, X_test_cv = X_a[train_index], X_a[test_index]
            y_train_cv, y_test_cv = y_a[train_index], y_a[test_index]
            
            # Train the model on the training split
            gp_model.fit(X_train_cv, y_train_cv)
            
            # Compute permutation importance on the test split
            perm_importance = permutation_importance(
                gp_model, X_test_cv, y_test_cv,
                n_repeats=num_repeat, random_state=random_state,
                scoring='neg_root_mean_squared_error'
            )
            
            # Accumulate the importances
            importances += perm_importance.importances_mean

        # Average the importances across all folds
        importances /= kf.get_n_splits()

    feature_names = X.columns.tolist()

    if fi_cv == False:
        perm_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': perm_importance.importances_mean})
    else:
        perm_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False,ignore_index=True)

    if weighted_fi == True:
        original_feature_names = []

        for col in X.columns:
            if "_" in col:
                original_feature = col.split("_")[0]
            else:
                original_feature = col  # For numeric or already unique features
            original_feature_names.append(original_feature)
        
        # Create a mapping from original feature names to their encoded columns
        from collections import defaultdict
        encoded_feature_map = defaultdict(list)

        for original_feature, encoded_feature in zip(original_feature_names, X.columns):
            encoded_feature_map[original_feature].append(encoded_feature)

        # Now, aggregate the importances using the generated mapping
        aggregated_importance = {}

        for original_feature, encoded_cols in encoded_feature_map.items():
            # Filter perm_importance_df for the relevant encoded columns
            encoded_importance = perm_importance_df[perm_importance_df['Feature'].isin(encoded_cols)]
            # Sum the importances for the encoded columns
            total_importance = encoded_importance['Importance'].sum()
            # Store the aggregated importance
            aggregated_importance[original_feature] = total_importance

        # Convert the aggregated importance dictionary to a DataFrame for easy viewing
        aggregated_importance_df = pd.DataFrame(list(aggregated_importance.items()), columns=['Feature', 'Aggregated Importance'])
        aggregated_importance_df = aggregated_importance_df.sort_values(by='Aggregated Importance', ascending=False, ignore_index=True)

        # Select the top 10 and bottom 10 features
        top_10 = aggregated_importance_df.head(10)
        bottom_10 = aggregated_importance_df.tail(10)

        # Concatinating and printing
        top_bottom_df = pd.concat([top_10, bottom_10], ignore_index=False)
        print(top_bottom_df)
        top_features = aggregated_importance_df.nlargest(top_bottom_features, 'Aggregated Importance')

    else:
        # Get the largest permutation importance values
        top_features = perm_importance_df.nlargest(top_bottom_features, 'Importance')

        # Get the smallest permutation importance values
        bottom_features = perm_importance_df.nsmallest(top_bottom_features, 'Importance').iloc[::-1]

        # Concatenate the two DataFrames
        combined_pf_df = pd.concat([top_features, bottom_features], axis=0)
        print(combined_pf_df)

' ======= Data Visualization ======= '
' --- Plotting covariance matrix --- '
if plot_covariance_matrix == True:
    covariance_matrix = gp_model.kernel_(X)
    # print(covariance_matrix)

    plt.figure(figsize=(8, 6))
    plt.imshow(covariance_matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Covariance Matrix")
    plt.show()

' -- Plotting feature importance '
if plot_feature_importance == True:
    # Excessive formatting:
    ssp_font = {'fontname':'Source Sans Pro'}
    subfig_label_fontsize = 30
    label_fs = 17 # 'xx-large'
    tick_fs = label_fs * 0.85 # 'x-large
    txt_size = label_fs * 0.65
    leg_txt_size = label_fs  * 0.7
    txt_spine_color = '#212121'  #'#212121' # 'black' #333333
    spine_thickness = 1.3

    plt.rcParams.update({
        'axes.titlesize': tick_fs,        # Title font size
        'axes.labelsize': label_fs,        # X and Y label font size
        'xtick.labelsize': tick_fs,       # X tick label font size
        'ytick.labelsize': tick_fs,       # Y tick label font size

        'axes.spines.top': False,    # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.linewidth': spine_thickness,        # Spine thickness
        'xtick.major.width': spine_thickness,     # Major tick thickness for x-axis
        'ytick.major.width': spine_thickness,     # Major tick thickness for y-axis
        'xtick.major.size': spine_thickness * 3,      # Major tick length for x-axis
        'ytick.major.size': spine_thickness * 3,      # Major tick length for y-axis
        
        'font.family': 'sans-serif',
        'font.sans-serif': ['Ubuntu'], # Futura, Source Sans Pro, Fira Sans, Roboto

        'axes.grid': True,
        'grid.alpha': 0.4, 
        'grid.linewidth': spine_thickness,

        'legend.fontsize': leg_txt_size,
        'legend.handletextpad': 0.4, 
        'legend.handlelength': 1.12,
        'legend.handleheight': 0.7,

        'text.color': txt_spine_color,                   # Color for all text
        'axes.labelcolor': txt_spine_color,              # Color for axis labels
        'axes.edgecolor': txt_spine_color,               # Color for axis spines
        'xtick.color': txt_spine_color,                  # Color for x-axis tick labels and ticks
        'ytick.color': txt_spine_color,                  # Color for y-axis tick labels and ticks
    })
    # Data formatting for plotting
    df = top_features.iloc[:features_to_plot]

    try:
        df['Importance'] = df['Importance'] * 1000
        importance = 'Importance'
    except:
        df['Aggregated Importance'] = df['Aggregated Importance'] * 1000
        importance = 'Aggregated Importance'

    # --- Plotting
    fig,ax = plt.subplots()

    # Apply the color function to the 'Feature' column in the dataframe
    bar_values = df[importance].values
    df['Bar_Color'] = df['Feature'].apply(get_bar_color)
    bar_colors = df['Bar_Color'].tolist()

    barplot = sns.barplot(data=df, x='Feature', y=importance, hue='Feature', 
                        palette=bar_colors, legend=False, edgecolor=txt_spine_color)
    
    # --- formatting and displaying
    ax.set_ylabel(r'Importance (mA/cm$^2$)',size=label_fs, labelpad=10)
    ax.set(xlabel=None)

    # - Setting up the legend:
    color_label_map = {
    '#009E73': 'Negatrode',
    '#E69F00': 'Electrolyte',
    '#56B4E9': 'Positrode',
    '#CC79A7': 'Co-sinter',
    '#F0E442': 'Testing'
    }
    unique_colors = df['Bar_Color'].unique()
    legend_patches = [mpatches.Patch(color=color, label=color_label_map[color]) for color in unique_colors if color in color_label_map]
    legend = ax.legend(handles=legend_patches, loc='upper right', fontsize=leg_txt_size)
    for handle in legend.legendHandles:
        handle.set_edgecolor(txt_spine_color)  # Set border color to black

    # - Modify the x-axis labels before applying LaTeX formatting
    current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    new_labels = []
    for label in df['Feature']:  # Use the original labels from the dataframe
        if label.startswith('Negatrode batch_'):
            # Extract date
            new_label = 'Negatrode batch_' + label[21:28]
            new_labels.append(new_label)
        elif label.startswith('Electrolyte spray batch'):
            new_label = label[:-1]
            new_labels.append(new_label)
        else:
            new_labels.append(label)

    ax.set_xticklabels(new_labels, rotation=50, ha='right', size=txt_size)

    ax.tick_params(axis='y', which='major', labelsize=tick_fs)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='x', which='major', pad=0)

    # Slight shift in xticklabels
    trans = mtrans.Affine2D().translate(20, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)

    sns.despine(trim=True) # trim=True

    plt.tight_layout()

    if save_fi is not None:
        fmat = save_fi.split('.', 1)[-1]
        fig.savefig(save_fi, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

' -- Plotting partial dependence plots '
if plot_pdp == True:
    # --- Plotting
    fig = plt.figure(figsize=(6.5, 5.5))

    # - Excessive formatting
    ssp_font = {'fontname':'Source Sans Pro'}
    subfig_label_fontsize = 30
    label_fs = 16 # 'xx-large'
    tick_fs = label_fs * 0.85 # 'x-large
    txt_size = label_fs * 0.7
    leg_txt_size = label_fs  * 0.6
    txt_spine_color = '#212121'  #'#212121' # 'black' #333333
    spine_thickness = 1.3

    plt.rcParams.update({
        'axes.titlesize': tick_fs,        # Title font size
        'axes.labelsize': label_fs,        # X and Y label font size
        'xtick.labelsize': tick_fs,       # X tick label font size
        'ytick.labelsize': tick_fs,       # Y tick label font size

        'axes.spines.top': False,    # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.linewidth': spine_thickness,        # Spine thickness
        'xtick.major.width': spine_thickness,     # Major tick thickness for x-axis
        'ytick.major.width': spine_thickness,     # Major tick thickness for y-axis
        'xtick.major.size': spine_thickness * 3,      # Major tick length for x-axis
        'ytick.major.size': spine_thickness * 3,      # Major tick length for y-axis
        
        'font.family': 'sans-serif',
        'font.sans-serif': ['Ubuntu'], # Futura, Source Sans Pro, Fira Sans, Roboto

        'legend.fontsize': leg_txt_size,
        'legend.handletextpad': 0.2,           

        'text.color': txt_spine_color,                   # Color for all text
        'axes.labelcolor': txt_spine_color,              # Color for axis labels
        'axes.edgecolor': txt_spine_color,               # Color for axis spines
        'xtick.color': txt_spine_color,                  # Color for x-axis tick labels and ticks
        'ytick.color': txt_spine_color,                  # Color for y-axis tick labels and ticks
    })

    # --- Initializing plot grid
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # - Setting axes
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    top_features = perm_importance_df.nlargest(4, 'Importance')['Feature'].tolist()

    # Plot partial dependence plots
    subfig_labels = ['a','b','c','d']

    for i, feature in enumerate(top_features):        
        display = PartialDependenceDisplay.from_estimator(gp_model, X_train, features=[feature], ax=axes[i]) #kind='both'

        letter = subfig_labels[i]
        axes[i].text(-0.04,1.01, letter, fontsize=subfig_label_fontsize,
                    ha='right',va='bottom',weight='bold',transform = axes[i].transAxes,**ssp_font) 
       
        if i == 0:
            plt.ylabel(r'Current (A/cm$^2$)') 
        else:
            plt.ylabel('')


    plt.subplots_adjust(top=0.93,hspace=0.5,wspace = 0.5) # hspace=0.5

    plt.tight_layout()

    plt.show()

' -- Plotting a specific PDP'
if plot_specific_pdp is not None:
    # --- Plotting
    fig, ax = plt.subplots()

    # - Excessive formatting
    ssp_font = {'fontname':'Source Sans Pro'}
    subfig_label_fontsize = 30
    label_fs = 24 # 'xx-large'
    tick_fs = label_fs * 0.85 # 'x-large
    txt_size = label_fs * 0.7
    leg_txt_size = label_fs  * 0.6
    txt_spine_color = '#212121'  #'#212121' # 'black' #333333
    spine_thickness = 1.3

    plt.rcParams.update({
        'axes.titlesize': tick_fs,        # Title font size
        'axes.labelsize': label_fs,        # X and Y label font size
        'xtick.labelsize': tick_fs,       # X tick label font size
        'ytick.labelsize': tick_fs,       # Y tick label font size

        'axes.spines.top': False,    # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.linewidth': spine_thickness,        # Spine thickness
        'xtick.major.width': spine_thickness,     # Major tick thickness for x-axis
        'ytick.major.width': spine_thickness,     # Major tick thickness for y-axis
        'xtick.major.size': spine_thickness * 3,      # Major tick length for x-axis
        'ytick.major.size': spine_thickness * 3,      # Major tick length for y-axis
        
        'font.family': 'sans-serif',
        'font.sans-serif': ['Ubuntu'], # Futura, Source Sans Pro, Fira Sans, Roboto

        'legend.fontsize': leg_txt_size,
        'legend.handletextpad': 0.2,           

        'text.color': txt_spine_color,                   # Color for all text
        'axes.labelcolor': txt_spine_color,              # Color for axis labels
        'axes.edgecolor': txt_spine_color,               # Color for axis spines
        'xtick.color': txt_spine_color,                  # Color for x-axis tick labels and ticks
        'ytick.color': txt_spine_color,                  # Color for y-axis tick labels and ticks
    })

    # Plot partial dependence plots
    feature = plot_specific_pdp

    display = PartialDependenceDisplay.from_estimator(gp_model, X_train, features=[feature], ax=ax,
                                                      line_kw={'color': '#CC4628', 'linewidth': spine_thickness*2})
       
    ax.set_ylabel(r'PPD (W/cm$^2$)') 

    plt.tight_layout()
    plt.show()

' Plotting Goodness of fit evaluators'
if all_gof == True:
    # - Running all folds and making dataframes
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state) # Initialize KFold
    
    residuals = [] # Arrays to store the residuals for each fold
    y_test_all = []  # To store all actual values
    y_pred_all = []  # To store all predicted values
    y_std_all = []
    ls_all = []

    # Ensure X and y are numpy arrays
    if isinstance(X, pd.DataFrame):
        X_gof = X.values
    if isinstance(y, pd.Series):
        y_gof = y.values
    if isinstance(ec_err, pd.Series):
        ec_err = ec_err.values

    # Perform 5-fold cross-validation
    for train_index, test_index in kf.split(X_gof):
        X_train, X_test = X_gof[train_index], X_gof[test_index]
        y_train, y_test = y_gof[train_index], y_gof[test_index]
        errors_train, errors_test = ec_err[train_index], ec_err[test_index]

        # Fit the model
        gp_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred_fold, y_std_fold = gp_model.predict(X_test, return_std=True)

        # Store actual and predicted values
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred_fold)
        y_std_all.extend(y_std_fold)

        # Calculate residuals
        fold_residuals = y_test - y_pred_fold
        residuals.extend(fold_residuals)

        # keeping track of lengthscales
        learned_length_scales = gp_model.kernel_.length_scale
        ls_all.append(learned_length_scales)

    # Convert lists to numpy arrays for easy plotting
    residuals = np.array(residuals)
    y_test_all = np.array(y_test_all)
    y_pred_all = np.array(y_pred_all)
    y_std_all = np.array(y_std_all)
    ls_all = np.array(ls_all)
    mean_length_scales = np.mean(ls_all, axis=0)


    # Calculate R^2 from residuals
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_test_all - np.mean(y_test_all))**2)
    r2 = 1 - (ss_res / ss_tot)
    r2_str = r'R$^2$: ' + f'{r2:.2f}'

    # --- Plotting
    fig = plt.figure(figsize=(11, 6.5))

    # - Excessive formatting
    ssp_font = {'fontname':'Source Sans Pro'}
    subfig_label_fontsize = 34
    label_fs = 21 # 'xx-large'
    tick_fs = label_fs * 0.85 # 'x-large
    txt_size = label_fs * 0.7
    leg_txt_size = label_fs  * 0.6
    txt_spine_color = '#212121'  #'#212121' # 'black' #333333
    spine_thickness = 1.3

    plt.rcParams.update({
        'axes.titlesize': tick_fs,        # Title font size
        'axes.labelsize': label_fs,        # X and Y label font size
        'xtick.labelsize': tick_fs,       # X tick label font size
        'ytick.labelsize': tick_fs,       # Y tick label font size

        'axes.spines.top': False,    # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.linewidth': spine_thickness,        # Spine thickness
        'xtick.major.width': spine_thickness,     # Major tick thickness for x-axis
        'ytick.major.width': spine_thickness,     # Major tick thickness for y-axis
        'xtick.major.size': spine_thickness * 3,      # Major tick length for x-axis
        'ytick.major.size': spine_thickness * 3,      # Major tick length for y-axis
        
        'font.family': 'sans-serif',
        'font.sans-serif': ['Ubuntu'], # Futura, Source Sans Pro, Fira Sans, Roboto

        'legend.fontsize': leg_txt_size,
        'legend.handletextpad': 0.2,           

        'text.color': txt_spine_color,                   # Color for all text
        'axes.labelcolor': txt_spine_color,              # Color for axis labels
        'axes.edgecolor': txt_spine_color,               # Color for axis spines
        'xtick.color': txt_spine_color,                  # Color for x-axis tick labels and ticks
        'ytick.color': txt_spine_color,                  # Color for y-axis tick labels and ticks
    })

    # --- Initializing plot grid
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])

    # - Setting axes
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 0:2])  # Span the first three columns
    ax5 = fig.add_subplot(gs[1, 2:6])  # Span the last three columns

    # --- Actual vs. predicted
    c_mkr = '#09396C' # light_blue: #879EC3 Blaster_blue: #09396C
    c_line = 'r'  # Colorado Red: '#CC4628' lighter tint: #d66a52
    ax1.scatter(y_test_all, y_pred_all, alpha=0.5,c=c_mkr)  # Create a scatterplot of actual vs. predicted values
    ax1.set_xlabel("Measured values")
    ax1.set_ylabel("Predicted values")
    ax1.text(-0.03,1.02, 'a', fontsize=subfig_label_fontsize,
                ha='right',va='bottom',weight='bold',transform =ax1.transAxes,**ssp_font)
    min_val = min(min(y_test_all), min(y_pred_all))
    max_val = max(max(y_test_all), max(y_pred_all))
    ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    
    # - Writing R^2 onto figure
    x0_range_ax1 = ax1.get_xlim()[1] - ax1.get_xlim()[0]
    y0_range_ax1 = ax1.get_ylim()[1] - ax5.get_ylim()[0]
    ax1.text(0.04,1.00,r2_str,weight='bold',size=tick_fs,
              ha='left', va='top', transform=ax1.transAxes)

    # --- Residuals vs. Predicted Values
    ax2.scatter(y_pred_all, residuals, alpha=0.5,c=c_mkr)
    ax2.axhline(y=0, color=c_line, linestyle='--')
    ax2.set_xlabel("Predicted values")
    ax2.set_ylabel("Residuals")
    ax2.text(-0.03,1.02, 'b', fontsize=subfig_label_fontsize,
                ha='right',va='bottom',weight='bold',transform =ax2.transAxes,**ssp_font)
    ax2.tick_params(axis='both', which='major')


    # --- Residuals Distribution
    sns.histplot(residuals, kde=True,stat="density",color=c_mkr,ax=ax3)
    sns.kdeplot(residuals,color=c_line,ax=ax3)
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Frequency")
    ax3.text(-0.03,1.02, 'c', fontsize=subfig_label_fontsize,
                ha='right',va='bottom',weight='bold',transform =ax3.transAxes,**ssp_font)
    ax3.tick_params(axis='both', which='major')

    # --- Learning curve plot:
    train_sizes, train_scores, validation_scores = learning_curve(gp_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
                                                                   cv=5, scoring='neg_root_mean_squared_error')

    # Calculate mean and standard deviation for training and validation scores
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    ax4.plot(train_sizes, validation_scores_mean, 'o-', color='#CC4628', label='Validation')
    ax4.plot(train_sizes, train_scores_mean, 'o-', color='#09396C', label='Training')
    ax4.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color='#CC4628')
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='#09396C')
    ax4.set_xlabel('Training set size')
    ax4.set_ylabel('RMSE (W/cm$^2$)')
    ax4.text(-0.03,1.02, 'd', fontsize=subfig_label_fontsize,
                 ha='right',va='bottom',weight='bold',transform =ax4.transAxes,**ssp_font)
    ax4.legend()

    # - Coverage plot:
    # - Misc formatting initializing
    if err == False:
        ax5.scatter(range(len(y_test_all)), y_test_all, color = '#09396C', label='Measured values')
        total_uncertainty = np.sqrt(y_std_all**2 + ec_err**2)
        coverage = ((y_test_all >= y_pred_all - 1.96 * total_uncertainty) & 
                    (y_test_all <= y_pred_all + 1.96 * total_uncertainty)).mean()
    elif err == True:
        ax5.errorbar(range(len(y_test_all)), y_test_all, yerr=ec_err, color = '#09396C', fmt='o', label='Actual values')
        coverage = ((y_test_all >= y_pred_all - 1.96 * y_std_all) & (y_test_all <= y_pred_all + 1.96 * y_std_all)).mean()

    ax5.errorbar(range(len(y_pred_all)), y_pred_all, yerr=1.96*y_std_all, fmt='o',color='#CC4628', label='Predictions with 95% CI')
            
    ax5.set_xlabel('Cell')
    ax5.set_ylabel(r'CD at 1.3 V (A/cm$^2$)')

    # - Printing figtext
    cov_str = f"Coverage probability: {coverage * 100:.1f}%"
    x0_center = np.mean(ax5.get_xlim())
    x0_range_ax5 = ax5.get_xlim()[1] - ax5.get_xlim()[0]
    y0_range_ax5 = ax5.get_ylim()[1] - ax5.get_ylim()[0]
    ax5.text(0.98, 1.0,cov_str,weight='bold',size=txt_size,
              ha='right', va='bottom',transform=ax5.transAxes)
    ax5.text(-0.03,1.02, 'e', fontsize=subfig_label_fontsize,
                 ha='right',va='bottom',weight='bold',transform=ax5.transAxes,**ssp_font)

    # - Excessive formatting:
    ax5.set_ylim(bottom=0)

    ax5.legend(loc='lower left', bbox_to_anchor=(0.0, 0.92))
    
    plt.subplots_adjust(hspace=0.5)

    plt.tight_layout()

    if save_gof is not None:
        fmat = save_gof.split('.', 1)[-1]
        fig.savefig(save_gof, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

    # --------- Calculations:
    # Calculate RMSE
    rmse_cv = np.sqrt(mean_squared_error(y_test_all, y_pred_all))

    # Calculate NLL
    nll_cv = calc_nll(y_test_all, y_pred_all, y_std_all, ec_err)

    # - Calculating MAE:
    mae = np.mean(np.abs(residuals))
    
    # - Print values
    print(f"RMSE: {rmse_cv:0.3f}")
    print(f"MAE: {mae:0.3f}")
    print(f"NLL: {nll_cv:0.3f}")
    print(r2_str)
