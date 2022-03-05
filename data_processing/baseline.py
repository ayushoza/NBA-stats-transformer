import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import ast

df = pd.read_csv("data_testing_standardized.csv", header=None)
test_numpy = df.to_numpy()
x_test = test_numpy[:, :-12]
shape_1 = test_numpy[:, -12:].shape
y_test = np.zeros(shape=(shape_1[0], shape_1[1], 13))
for i in range(shape_1[0]):
    convert = []
    for j in range(shape_1[1]):
        s = " ".join(test_numpy[i][-12 + j].split()).replace(" ", ",").replace(
            "[,", "[").replace(",]", "]").replace("nan", "0")
        convert.append(np.array(ast.literal_eval(s), dtype=object))
    y_test[i] = np.array([convert])

preds = np.zeros(shape=(len(test_numpy[:, -13]), 12, 13))
for i in range(len(test_numpy[:, -13])):
    s = " ".join(test_numpy[i][-13].split()).replace(" ", ",").replace(
        "[,", "[").replace(",]", "]").replace("nan", "0")
    convert = np.array(ast.literal_eval(s), dtype=object)
    preds[i] = np.array([convert] * 12)


def evaluate_baseline(preds, y_test):
    """
    Uses the last month approach as a baseline taking in predictions (all of the previous
    month statistics) and y_test (all labels for 12 months to follow) and calculates 
    mean squared error for each statistic.
    """
    stats_mse = []
    for i in range(12):
        stat_preds = preds[:, :, i]
        stat_labels = y_test[:, :, i]
        stats_mse.append(mean_squared_error(stat_labels, stat_preds))
    return stats_mse


mses = evaluate_baseline(preds, y_test)
print(mses)
print("mean MSE for all stats = ", np.mean(mses))
