from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from utils_csv_to_list import merged_csv_to_list, read_csv_to_list
from element_binary_to_proportion import binary_to_proportion

input_features = binary_to_proportion(read_csv_to_list("populations.csv"))
file_paths = 'fitness.csv'
output_labels = np.array(merged_csv_to_list(file_paths), dtype=np.float32)

rf = RandomForestRegressor()

param_grid = {
    'n_estimators': [25, 50, 100, 150 ,200 ,300 ,400],
    'max_depth': [None, 10 ,20 ,30 ,40,50],
    'min_samples_split': [2, 3 ,4 ,5 ,6,8, 10, 15, 20],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

kf = KFold(n_splits=10, shuffle=True)
best_r2 = -float("inf")
best_mse = float("inf")
best_params = None
best_fold = None
best_model = None

for i, params in enumerate(ParameterGrid(param_grid)):

    rf.set_params(**params)

    for j, (train_index, val_index) in enumerate(kf.split(input_features)):
        input_train, input_val = input_features[train_index], input_features[val_index]
        output_train, output_val = output_labels[train_index], output_labels[val_index]

        rf.fit(input_train, output_train)

        y_pred = rf.predict(input_val)

        r2 = r2_score(output_val, y_pred)
        mse = mean_squared_error(output_val, np.clip(y_pred, 0, 1))

        params_str = str(params).replace(" ", "").replace(":", "_").replace("'", "").replace("{", "").replace("}", "")

        filename = f"fold{j}_{r2:.4f}_{mse:.4f}.joblib"

        joblib.dump(rf, filename)
        print(f"save as: {filename}，parameter: {params}，fold_num: {j}")
        if (r2 > best_r2) or (np.isclose(r2, best_r2) and mse < best_mse):
            best_r2 = r2
            best_mse = mse
            best_params = params.copy()
            best_fold = j
            best_model = rf

if best_model is not None:
    best_filename = f"best_fold{best_fold}_{best_r2:.4f}_{best_mse:.4f}.joblib"
    joblib.dump(best_model, best_filename)