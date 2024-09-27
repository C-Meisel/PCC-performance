'''
This script is used to implement a Random Forest Regressor (RFR) model of PCC performance data
The model can analyze PPD data gathered from 4411 cells tested at Mines
Written with the help of ChatGPT
C-Meisel
'''

' Imports '
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtrans
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.manifold import MDS
import os
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text, plot_tree
from sklearn import tree


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

# - Misc functions:
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
                          'Co-sinter batch','Absolute humidity at co-sinter (g/m^3)','Sintering neighbor', 'Days (Co-sinter to test)']
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

' --- Cols to Edit:'
feature_importance = False
feature_importance_cv = True
features_to_plot = 10
plot_best_tree = False
mds_plot = False

' - Plotting '
plot_feature_importance =True
plot_pdp = True
all_gof = True

pdps = 5
plot_specific_pdp = None #'Absolute humidity at co-sinter (g/m^3)'

n_est = 99
max_feat= 0.205
mss = 2 # min_samples_split
msl= 1 # min_samples_leaf
ccp_alpha = 0
rs = 76
cv = 5

' - Saving figures - '
save_folder_loc = 'Location of the folder where the graphs'

save_fi = None # os.path.join(save_folder_loc,'RFR_FI_FC_figure_name.png')
save_pdp =  None # os.path.join(save_folder_loc,'RFR_PDP_FC_figure_name.png')
save_gof = None #os.path.join(save_folder_loc,'RFR_GOF_FC_figure_name.png')

' ------- Loading Data - '
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

print('Number of parameters in the model: ',X.shape[1])


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

print('Number of encoded parameters in the model: ',X.shape[1])


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/cv, random_state=rs)

# Initialize the Random Forest classifier
rfr = RandomForestRegressor(n_estimators=n_est, max_features = max_feat, min_samples_split = mss, min_samples_leaf = msl,
                             random_state=rs, oob_score = True)

# Train the model
kf = KFold(n_splits=cv, shuffle=True, random_state=rs)
rfr.fit(X_train, y_train)

' === Feature importances '
if feature_importance == True:
    # Get the feature names from the preprocessor
    feature_names = X.columns.tolist()

    perm_importance = permutation_importance(rfr, X_test, y_test,
                                            n_repeats=3, random_state=rs, scoring='neg_root_mean_squared_error')
    perm_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': perm_importance.importances_mean})
    perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False,ignore_index=True)


    # Get the largest permutation importance values
    top_features = perm_importance_df.nlargest(10, 'Importance')

    # Get the smallest permutation importance values
    bottom_features = perm_importance_df.nsmallest(10, 'Importance').iloc[::-1]

    # Concatenate the two DataFrames
    combined_pf_df = pd.concat([top_features, bottom_features], axis=0)
    print(combined_pf_df)

if feature_importance_cv == True:
        importances = np.zeros(X.shape[1])

        # Iterate through the KFold splits
        # Ensure X and y are numpy arrays
        if isinstance(X, pd.DataFrame):
            X_a = X.values
        if isinstance(y, pd.Series):
            y_a = y.values
        if isinstance(ppd_err, pd.Series):
            ppd_err = ppd_err.values

        for train_index, test_index in kf.split(X_a):
            X_train_cv, X_test_cv = X_a[train_index], X_a[test_index]
            y_train_cv, y_test_cv = y_a[train_index], y_a[test_index]
            
            # Train the model on the training split
            rfr.fit(X_train_cv, y_train_cv)
            
            # Compute permutation importance on the test split
            perm_importance = permutation_importance(
                rfr, X_test_cv, y_test_cv,
                n_repeats=3, random_state=rs,
                scoring='neg_root_mean_squared_error'
            )
            
            # Accumulate the importances
            importances += perm_importance.importances_mean

        # Average the importances across all folds
        importances /= kf.get_n_splits()

        feature_names = X.columns.tolist()

        perm_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False,ignore_index=True)

        # Get the largest permutation importance values
        top_features = perm_importance_df.nlargest(10, 'Importance')

        # Get the smallest permutation importance values
        bottom_features = perm_importance_df.nsmallest(10, 'Importance').iloc[::-1]

        # Concatenate the two DataFrames
        combined_pf_df = pd.concat([top_features, bottom_features], axis=0)
        print(combined_pf_df)

' ======= Data Visualization ======= '
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
    df['Importance'] = df['Importance'] * 1000

    # --- Plotting
    fig,ax = plt.subplots()

    # Apply the color function to the 'Feature' column in the dataframe
    bar_values = df['Importance'].values
    df['Bar_Color'] = df['Feature'].apply(get_bar_color)
    bar_colors = df['Bar_Color'].tolist()

    barplot = sns.barplot(data=df, x='Feature', y='Importance', hue='Feature', 
                        palette=bar_colors, legend=False, edgecolor=txt_spine_color)
    
    # --- formatting and displaying
    ax.set_ylabel(r'Importance (mW/cm$^2$)',size=label_fs, labelpad=10)
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
        elif 'batch_' in label:
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
    # - Excessive formatting
    ssp_font = {'fontname':'Source Sans Pro'}
    subfig_label_fontsize = 34
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

        'axes.grid': True,
        'grid.alpha': 0.4, 
        'grid.linewidth': spine_thickness,

        'legend.fontsize': leg_txt_size,
        'legend.handletextpad': 0.2,           

        'text.color': txt_spine_color,                   # Color for all text
        'axes.labelcolor': txt_spine_color,              # Color for axis labels
        'axes.edgecolor': txt_spine_color,               # Color for axis spines
        'xtick.color': txt_spine_color,                  # Color for x-axis tick labels and ticks
        'ytick.color': txt_spine_color,                  # Color for y-axis tick labels and ticks
    })

    # Filter out catagorical features
    filtered_features_df = perm_importance_df[~perm_importance_df['Feature'].str.contains('_')]

    if pdps == 3:
        fig = plt.figure(figsize=(10, 3.5))

        gs = gridspec.GridSpec(1, 3)

        axes = [fig.add_subplot(gs[i, j]) for i in range(1) for j in range(3)]

        # Select the top 3 features based on importance
        top_features = filtered_features_df.nlargest(3, 'Importance')['Feature'].tolist()
            
        subfig_labels = ['a','b','c']

        for i, feature in enumerate(top_features):  
            color = 'forestgreen' # '#00341C'
            ax = axes[i]

            display = PartialDependenceDisplay.from_estimator(rfr, X_train, features=[feature],
                                                               ax=ax,line_kw={'color': color, 'linewidth': spine_thickness*2}) #kind='both'

            letter = subfig_labels[i]
            ax.text(-0.04,1.01, letter, fontsize=subfig_label_fontsize,
                        ha='right',va='bottom',weight='bold',transform = ax.transAxes,**ssp_font) 
        
            if i == 0:
                plt.ylabel(r'PPD (W/cm$^2$)') 
            else:
                plt.ylabel('')

            if feature == 'Electrolyte thickness to grain size ratio':
                plt.xlabel('Thickness : Grain size')


    elif pdps == 4:
        fig = plt.figure(figsize=(6.5, 5.5))

        # --- Initializing plot grid
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

        # - Setting axes
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

        top_features = filtered_features_df.nlargest(pdps, 'Importance')['Feature'].tolist()

        # Plot partial dependence plots
        subfig_labels = ['a','b','c','d']

        for i, feature in enumerate(top_features):  
            color = 'forestgreen' # '#00341C'
            ax = axes[i]

            display = PartialDependenceDisplay.from_estimator(rfr, X_train, features=[feature],
                                                               ax=ax,line_kw={'color': color, 'linewidth': spine_thickness*2}) #kind='both'

            letter = subfig_labels[i]
            ax.text(-0.04,1.01, letter, fontsize=subfig_label_fontsize,
                        ha='right',va='bottom',weight='bold',transform = ax.transAxes,**ssp_font) 
        
            if i == 0:
                plt.ylabel(r'PPD (W/cm$^2$)') 
            else:
                plt.ylabel('')

            if feature == 'Electrolyte thickness to grain size ratio':
                plt.xlabel('Thickness : Grain size')

    else:
        fig = plt.figure(figsize=(11, 6.5))

        # --- Initializing plot grid
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

        # - Setting axes
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

        top_features = filtered_features_df.nlargest(pdps, 'Importance')['Feature'].tolist()

        # Plot partial dependence plots
        subfig_labels = ['a','b','c','d','e','f']

        for i, feature in enumerate(top_features):  
            color = 'forestgreen' # '#00341C'
            ax = axes[i]

            display = PartialDependenceDisplay.from_estimator(rfr, X_train, features=[feature],
                                                               ax=ax,line_kw={'color': color, 'linewidth': spine_thickness*2}) #kind='both'

            letter = subfig_labels[i]
            ax.text(-0.04,1.01, letter, fontsize=subfig_label_fontsize,
                        ha='right',va='bottom',weight='bold',transform = ax.transAxes,**ssp_font) 
        
            if i == 0 or 3:
                plt.ylabel(r'PPD (W/cm$^2$)') 
            else:
                plt.ylabel('')

            if feature == 'Electrolyte thickness to grain size ratio':
                plt.xlabel('Thickness : Grain size')
        
        if pdps == 5:
            fig.delaxes(axes[pdps])


    plt.subplots_adjust(top=0.93,hspace=0.5,wspace = 0.5) # hspace=0.5

    plt.tight_layout()

    if save_pdp is not None:
        fmat = save_pdp.split('.', 1)[-1]
        fig.savefig(save_pdp, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

if plot_specific_pdp is not None:
    # --- Plotting
    fig, ax = plt.subplots()

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

    feature = plot_specific_pdp

    display = PartialDependenceDisplay.from_estimator(rfr, X_train, features=[feature], ax=ax)
       
    ax.set_ylabel(r'PPD (W/cm$^2$)') 

    plt.tight_layout()
    plt.show()

' Plotting Goodness of fit evaluators'
df_X = X
if all_gof == True:
    # - Running all folds and making dataframes    
    residuals = [] # Arrays to store the residuals for each fold
    y_test_all = []  # To store all actual values
    y_pred_all = []  # To store all predicted values
    rmse_all = []
    all_shap_values = []
    all_predictions = []
    oob_scores = []

    # Ensure X and y are numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(ppd_err, pd.Series):
        ppd_err = ppd_err.values

    # Perform 5-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        errors_train, errors_test = ppd_err[train_index], ppd_err[test_index]

        rfr.fit(X_train, y_train)

        # Predict on the test set
        y_pred_fold = rfr.predict(X_test)

        # Store actual and predicted values
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred_fold)

        # Calculate residuals
        fold_residuals = y_test - y_pred_fold
        residuals.extend(fold_residuals)

        # - Calculate rmse:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_fold))
        rmse_all.append(rmse)

        # - Save OOB score:
        oob_scores.append(rfr.oob_score_)

    # Convert lists to numpy arrays for easy plotting
    residuals = np.array(residuals)
    y_test_all = np.array(y_test_all)
    y_pred_all = np.array(y_pred_all)
    rmse_all = np.array(rmse_all)

    # Calculate R^2 from residuals
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_test_all - np.mean(y_test_all))**2)
    r2 = 1 - (ss_res / ss_tot)
    r2_str = r'R$^2$: ' + f'{r2:.2f}'

    # --- Plotting
    fig = plt.figure(figsize=(7, 6.4))

    # - Excessive formatting
    ssp_font = {'fontname':'Source Sans Pro'}
    subfig_label_fontsize = 30
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

        'axes.grid': True,
        'grid.alpha': 0.4, 
        'grid.linewidth': spine_thickness,
        
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
    # gs = gridspec.GridSpec(3, 6, height_ratios=[1, 0.01, 1])
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # - Setting axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])  # Span the first three columns

    # --- Actual vs. predicted
    c_mkr = '#09396C' # light_blue: #879EC3 Blaster_blue: #09396C
    c_line = 'r'  # Colorado Red: '#CC4628' lighter tint: #d66a52
    ax1.scatter(y_test_all, y_pred_all, alpha=0.5,c=c_mkr)  # Create a scatterplot of actual vs. predicted values
    ax1.set_xlabel("Measured values")
    ax1.set_ylabel("Predicted values")
    ax1.text(-0.04,1.01, 'a', fontsize=subfig_label_fontsize,
                 ha='right',va='bottom',weight='bold',transform =ax1.transAxes,**ssp_font)    
    min_val = min(min(y_test_all), min(y_pred_all))
    max_val = max(max(y_test_all), max(y_pred_all))
    ax1.plot([min_val, max_val], [min_val, max_val], color=c_line, linestyle='--')
    
    # - Writing R^2 onto figure
    x0_range_ax1 = ax1.get_xlim()[1] - ax1.get_xlim()[0]
    y0_range_ax1 = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    ax1.text(0.04,1.00,r2_str,weight='bold',size=tick_fs,
              ha='left', va='top', transform=ax1.transAxes)

    # --- Residuals vs. Predicted Values
    ax2.scatter(y_pred_all, residuals, alpha=0.5,c=c_mkr)
    ax2.axhline(y=0, color=c_line, linestyle='--')
    ax2.set_xlabel("Predicted values")
    ax2.set_ylabel("Residuals")
    ax2.text(-0.04,1.01, 'b', fontsize=subfig_label_fontsize,
                 ha='right',va='bottom',weight='bold',transform =ax2.transAxes,**ssp_font)
    ax2.tick_params(axis='both', which='major')


    # --- Residuals Distribution
    sns.histplot(residuals, kde=True,stat="density",color=c_mkr,ax=ax3)
    sns.kdeplot(residuals,color=c_line,ax=ax3)
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Frequency")
    ax3.text(-0.04,1.01, 'c', fontsize=subfig_label_fontsize,
                 ha='right',va='bottom',weight='bold',transform =ax3.transAxes,**ssp_font)
    ax3.tick_params(axis='both', which='major')
    ax3.grid(axis='x')

    # --- Learning curve plot:
    train_sizes, train_scores, validation_scores = learning_curve(
        rfr, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=4, scoring='neg_root_mean_squared_error'
    )

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
    ax4.text(-0.04,1.01, 'd', fontsize=subfig_label_fontsize,
                 ha='right',va='bottom',weight='bold',transform =ax4.transAxes,**ssp_font)
    ax4.legend()

    # - Printing values
    rmse_cv = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
    mae = np.mean(np.abs(residuals))
    mean_oob_score = np.mean(oob_scores)

    print(f"RMSE: {rmse_cv:0.3f}")
    print(f"MAE: {mae:0.3f}")
    print(f'OOB Score: {mean_oob_score:.2f}')
    print(r2_str)

    plt.tight_layout()

    plt.subplots_adjust(top=0.92,hspace=0.5,wspace = 0.4) # hspace=0.5

    if save_gof is not None:
        fmat = save_gof.split('.', 1)[-1]
        fig.savefig(save_gof, dpi=300, format=fmat, bbox_inches='tight')

    plt.show()

' ----- Plotting the best decision tree '
if plot_best_tree == True:
    feature_names=X.columns.tolist()

    tree_performance = []

    for i, est in enumerate(rfr.estimators_):
        # Predict using the individual tree
        y_pred = est.predict(X_test)
        
        # Calculate mean squared error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        tree_performance.append((i, rmse))

    # Find the best tree
    best_tree_index, best_tree_rmse = min(tree_performance, key=lambda x: x[1])
    best_tree = rfr.estimators_[best_tree_index]

    print(f'Best tree index: {best_tree_index}, RMSE: {best_tree_rmse}')

    plt.figure(figsize=(20,10))
    tree.plot_tree(best_tree, 
                feature_names=feature_names, 
                filled=True, 
                rounded=True, 
                fontsize=10)
    plt.show()

    tree_rules = export_text(best_tree, feature_names=feature_names)
    print(tree_rules)

if mds_plot == True:
    proximity_matrix = np.zeros((X.shape[0], X.shape[0]))

    # Loop through each tree in the forest
    for tree in rfr.estimators_:
        # Get the leaf indices for each sample
        leaf_indices = tree.apply(X)

        # Loop through each sample and update the proximity matrix
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if leaf_indices[i] == leaf_indices[j]:
                    proximity_matrix[i, j] += 1

    # Normalize the proximity matrix by the number of trees
    proximity_matrix /= len(rfr.estimators_)

    # Convert the proximity matrix to a distance matrix
    distance_matrix = 1 - proximity_matrix

    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_results = mds.fit_transform(distance_matrix)

    # Plot the MDS results
    plt.figure(figsize=(10, 8))
    plt.scatter(mds_results[:, 0], mds_results[:, 1], c=y, cmap='viridis', s=50)
    plt.colorbar(label='Target Value')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.title('MDS Plot using Random Forest Proximity Matrix')
    plt.show()

    # Convert MDS results to a DataFrame
    mds_df = pd.DataFrame(mds_results, columns=['MDS1', 'MDS2'])

    # Concatenate original features and MDS dimensions
    combined_df = pd.concat([X, mds_df], axis=1)

    # Compute the correlation matrix
    correlation_matrix = combined_df.corr()

    # Extract the correlations of original features with MDS dimensions
    mds_correlations = correlation_matrix[['MDS1', 'MDS2']].loc[X.columns]

    # Display the correlation matrix
    print(mds_correlations)

    # Sort features by their correlation with MDS1
    mds1_correlations_sorted = mds_correlations['MDS1'].sort_values(ascending=False) # Ascending = True for bottom10. .abs() for absolute values
    top_10_features = mds1_correlations_sorted.head(10)
    print(top_10_features)

    mds2_correlations_sorted = mds_correlations['MDS2'].sort_values(ascending=True)
    bottom_10_mds2 = mds2_correlations_sorted.head(10)
    print(bottom_10_mds2)

    # Plot the correlation matrix
    plt.figure(figsize=(12, 6))
    plt.imshow(mds_correlations, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(mds_correlations.columns)), mds_correlations.columns, rotation=90)
    plt.yticks(range(len(mds_correlations.index)), mds_correlations.index)
    plt.title('Correlation of Original Features with MDS Dimensions')
    plt.show()


    plt.figure(figsize=(10, 6))
    top_10_features.plot(kind='bar')
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation with MDS1')
    plt.title('Top 10 Features Affecting MDS Dimension 1')
    plt.tight_layout()
    plt.show()
