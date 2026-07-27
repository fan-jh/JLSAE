import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def construction_error(Y, X):
    '''
    param Y:Original HSI
    param X:reconstructed background
    return: anomaly score
    '''

    res = Y - X
    y_score = np.linalg.norm(res, axis=2)
    y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))

    # Alternative: calculate the anomaly score using the Mahalanobis distance.
    # In most cases, the L2 norm above is sufficient. For a few data sets,
    # the Mahalanobis distance may provide better detection performance.
    #
    # res = np.asarray(Y - X, dtype=np.float64)
    # W, H, B = res.shape
    #
    # # Reshape the residual cube into a two-dimensional matrix
    # res_2d = res.reshape(W * H, B)
    #
    # # Remove the mean residual spectrum
    # mu = np.mean(res_2d, axis=0, keepdims=True)
    # res_centered = res_2d - mu
    #
    # # Estimate the covariance matrix
    # S = (res_centered.T @ res_centered) / (W * H)
    #
    # # A small regularization term can be added to improve numerical stability
    # delta = 1e-6 # S = S + delta * np.eye(B)
    #
    # # Compute the squared Mahalanobis distance of each pixel
    # S_inv = np.linalg.pinv(S)
    # y_score = np.sum((res_centered @ S_inv) * res_centered, axis=1)
    # y_score = y_score.reshape(W, H)
    #
    # # Normalize the anomaly scores into [0, 1]
    # y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))

    return y_score

def calculate_fpr_and_fpr(y_true, y_score):
    '''Calculate the ROC curve using NumPy arrays.
    param y_true: Ground-truth labels
    param y_score: Estimated anomaly scores
    '''
    dim_y_true = len(y_true.shape)
    dim_y_score = len(y_score.shape)
    if dim_y_score == 2:
        y_true = y_true.flatten()
    if dim_y_true == 2:
        y_score = y_score.flatten()

    # Count the numbers of positive and negative samples
    p = y_true.sum()
    n = len(y_true) - p

    tp, fp = 0, 0
    tpr, fpr, thresholds = [], [], []
    score = max(y_score) + 1
    for i in np.flip(np.argsort(y_score)):
        if score != y_score[i]:
            tpr.append(tp/p)
            fpr.append(fp/n)
            thresholds.append(score)
            score = y_score[i]
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
    tpr.append(tp / p)
    fpr.append(fp / n)
    thresholds.append(score)

    return fpr, tpr, thresholds


def draw_roc(fpr, tpr):
    # Plot the ROC curve
    plt.plot(fpr, tpr)
    plt.axis("square")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.show()


def calculate_auc(fpr, tpr, method='trapezoid'):
    '''
    param fpr: False positive rate
    param tpr: True positive rate
    param method: Numerical integration method
    return: AUC value
    '''

    assert method in ['trapezoid', 'simpson'], "The integration method should be either 'trapezoid' or 'simpson'."
    if method == 'trapezoid':
        return np.trapz(tpr, fpr)
    else:
        return simps(tpr, fpr)








