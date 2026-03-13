import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import warnings

# Import the complexity module from the same directory
from .Complexity_metrics_algorithm import compute_complexity_metrics

warnings.filterwarnings('ignore')

def topological_ratio_optimizer(X, y, theta=0.05, step=0.05, random_state=42):
    """
    Generalization of HARS using topological complexity metrics.
    Finds the optimal sampling ratio where the vulnerability of both classes is balanced.
    
    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target vector (numpy array or pandas Series).
    - theta: Tolerance threshold for complexity difference (|C_min - C_maj|).
    - step: Ratio increment for each iteration.
    - random_state: Seed for reproducibility.
    
    Returns:
    - best_ratio: The optimal sampling ratio (float).
    - history_df: DataFrame containing the metrics per iteration for analysis.
    """
    
    # 1. Identify classes and initial class distribution
    classes, counts = np.unique(y, return_counts=True)
    min_class = classes[np.argmin(counts)]
    maj_class = classes[np.argmax(counts)]
    
    n_min = counts[np.argmin(counts)]
    n_maj = counts[np.argmax(counts)]
    original_ratio = n_min / n_maj
    
    # 2. Initial Evaluation (Baseline Topology)
    met_min_orig = compute_complexity_metrics(X, y, target_class=min_class)
    met_maj_orig = compute_complexity_metrics(X, y, target_class=maj_class)
    
    c_min_val = met_min_orig['dwCM9']
    c_maj_val = met_maj_orig['dwCM9']
    initial_diff = abs(c_min_val - c_maj_val)
    
    # Early stop if classes are already topologically balanced
    if initial_diff <= theta:
        return original_ratio, None

    # 3. Iterative Search for Optimal Ratio (Oversampling)
    ratios_to_test = np.arange(original_ratio + step, 1.0, step)
    ratios_to_test = np.append(ratios_to_test, 1.0) # Ensure 1.0 is tested
    
    best_ratio = 1.0
    min_difference = float('inf')
    iteration_history = []
    
    for r in ratios_to_test:
        # Apply synthetic oversampling for current ratio
        smote = SMOTE(sampling_strategy=r, random_state=random_state)
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
        except ValueError:
            # Handle edge cases where SMOTE fails (e.g., too few neighbors)
            continue
        
        # Recalculate topological complexity in the modified space
        met_min = compute_complexity_metrics(X_resampled, y_resampled, target_class=min_class)
        met_maj = compute_complexity_metrics(X_resampled, y_resampled, target_class=maj_class)
        
        current_diff = abs(met_min['dwCM9'] - met_maj['dwCM9'])
        
        iteration_history.append({
            'Ratio': round(r, 3), 
            'C_min_dwCM9': met_min['dwCM9'], 
            'C_maj_dwCM9': met_maj['dwCM9'], 
            'Absolute_Difference': current_diff
        })
        
        # Update best ratio found
        if current_diff < min_difference:
            min_difference = current_diff
            best_ratio = r
            
        # Stopping criterion: Intersection of vulnerabilities
        if current_diff <= theta:
            best_ratio = r
            break

    history_df = pd.DataFrame(iteration_history)
    
    return best_ratio, history_df
