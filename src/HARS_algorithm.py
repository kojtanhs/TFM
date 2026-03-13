
import statistics
from numpy import *
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTEENN, SMOTETomek

#from Algorithm_code.Hostility_measure_algorithm import hostility_measure
from Hostility_measure_algorithm import hostility_measure


def SM_r(sampling_method, ratio):

    if (sampling_method == 'RUS'):
        # Undersampling
        SM = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
    elif (sampling_method == 'ROS'):
        # Oversampling
        SM = RandomOverSampler(sampling_strategy=ratio, random_state=0)
    elif (sampling_method == 'ADASYN'):
        # ADASYN
        SM = ADASYN(sampling_strategy=ratio, random_state=0)
    elif (sampling_method == 'ClusterCentroids'):
        # ClusterCentroids
        SM = ClusterCentroids(sampling_strategy=ratio, random_state=0)
    elif (sampling_method == 'SMOTE'):
        # SMOTE
        SM = SMOTE(sampling_strategy=ratio, random_state=0)
    elif (sampling_method == 'SMOTEENN'):
        # SMOTEENN
        SM = SMOTEENN(sampling_strategy=ratio, random_state=0)
    elif (sampling_method == 'SMOTETomek'):
        # SMOTETomek
        SM = SMOTETomek(sampling_strategy=ratio, random_state=0)
    else:
        raise ValueError("The sampling method is not yet included.")
    return SM


def HARS(X_train,y_train,theta,sampling_method,ratio_v):
    """
    :param X_train: instances
    :param y_train: labels
    :param theta: the threshold to check the difference and the mean in hostility per classes
    :param sampling_method: sampling method selected by the user
    :param ratio_v: list of sampling ratios to test
    :return: ratio - best sampling ratio
    """

    # Check initial ratios
    n0 = len(y_train[y_train==0])
    n1 = len(y_train[y_train == 1])
    original_ratio = n1/n0
    ratios_no0 = list((x for x in ratio_v if x>0))
    if len(list((x for x in ratios_no0 if x <= original_ratio)))>0:
        raise ValueError("The considered ratios cannot be lower than the original one: ",original_ratio)

    # Check if 0 (i.e. no sampling) is in the ratio list
    if (0 not in ratio_v):
        ratio_v.insert(0, 0)

    # Initial values
    found = False
    i=0
    h0_v = [0] * len(ratio_v)
    h1_v = [0] * len(ratio_v)

    while ((not found) and (i < len(ratio_v))):
        ratio = ratio_v[i]
        if (ratio == 0): # no sampling
            X_train_r, y_train_r = X_train, y_train
        else:
            SM = SM_r(sampling_method,ratio)
            X_train_r, y_train_r = SM.fit_resample(X_train, y_train)

        # Hostility
        _, _, results_host, k_auto = hostility_measure(X_train_r, y_train_r, sigma=5, delta=0.5, k_min=0, seed=0)
        results_k = results_host.loc[k_auto, :]
        h0 = float(results_k.iloc[2])
        h1 = float(results_k.iloc[3])
        h0_v[i] = h0
        h1_v[i] = h1

        if ((abs(h0-h1)<theta) or (statistics.mean([h0,h1])<theta)):
            found = True
        else:
            i = i+1
    if found:
        ratio = ratio_v[i]
    else: # the conditions are never fulfilled
        best_i = np.argmin(abs(np.array(h0_v) - np.array(h1_v)))
        ratio = ratio_v[best_i]

    return ratio

# ratio = HARS(X_train,y_train,theta,sampling_method,ratio_v)

