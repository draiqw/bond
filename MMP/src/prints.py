from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def print_metrics(y_test_buy, y_pred_buy, name):
    print(f"Regression metrics for {name} prediction:")
    print("MAE:", mean_absolute_error(y_test_buy, y_pred_buy))
    print("MSE:", mean_squared_error(y_test_buy, y_pred_buy))
    print("R²:", r2_score(y_test_buy, y_pred_buy))
    print()

import pandas as pd

def print_feature_importance(model, feature_names):
    importance = pd.Series(model.coef_, index=feature_names)
    importance = importance.sort_values(key=abs, ascending=False)  # по модулю

    print("Feature importance (по абсолютной величине коэффициентов):")
    print(importance)
    return importance