import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def compute_complexity_metrics(X, y, target_class=1, k_neighbors=5):
    """
    Computes geometric complexity metrics (N3, L2, dwCM9) for a specific target class.
    Focuses on measuring topological vulnerability and class overlap.
    
    Parameters:
    - X: Feature matrix (numpy array).
    - y: Target vector (numpy array).
    - target_class: The class label to evaluate (int).
    - k_neighbors: Number of neighbors for k-NN based metrics (int).
    
    Returns:
    - Dictionary with N3, L2, and dwCM9 values.
    """
    
    # Identify instances of the target class and the opposite class
    target_idx = np.where(y == target_class)[0]
    opposite_idx = np.where(y != target_class)[0]
    
    X_target = X[target_idx]
    X_opposite = X[opposite_idx]
    
    n_target = len(X_target)
    
    if n_target == 0:
        return {'N3': 0.0, 'L2': 0.0, 'dwCM9': 0.0}

    # Scale data for distance calculations and SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_target_scaled = X_scaled[target_idx]
    
    # ----------------------------------------------------
    # 1. Calculate N3 (Error rate of 1-NN)
    # Measures local overlap
    # ----------------------------------------------------
    nn_1 = NearestNeighbors(n_neighbors=2) # 2 because the point itself is the nearest
    nn_1.fit(X_scaled)
    distances, indices = nn_1.kneighbors(X_target_scaled)
    
    # Check if the nearest neighbor (index 1) belongs to the opposite class
    nearest_neighbor_labels = y[indices[:, 1]]
    n3_errors = np.sum(nearest_neighbor_labels != target_class)
    n3_value = n3_errors / n_target

    # ----------------------------------------------------
    # 2. Calculate L2 (Error rate of Linear SVM)
    # Measures linear inseparability
    # ----------------------------------------------------
    svm = LinearSVC(random_state=42, dual=False)
    svm.fit(X_scaled, y)
    svm_predictions = svm.predict(X_target_scaled)
    
    l2_errors = np.sum(svm_predictions != target_class)
    l2_value = l2_errors / n_target

    # ----------------------------------------------------
    # 3. Calculate dwCM9 (Density-Weighted Complexity)
    # Measures topological vulnerability based on local density
    # ----------------------------------------------------
    k = min(k_neighbors, len(y) - 1)
    nn_k = NearestNeighbors(n_neighbors=k + 1)
    nn_k.fit(X_scaled)
    
    _, k_indices = nn_k.kneighbors(X_target_scaled)
    
    vulnerability_scores = []
    
    # Evaluate local neighborhood for each target instance
    for neighbors in k_indices:
        neighbor_labels = y[neighbors[1:]] # Exclude self
        # Count how many neighbors belong to the opposite class
        opposite_count = np.sum(neighbor_labels != target_class)
        ratio_opposite = opposite_count / k
        vulnerability_scores.append(ratio_opposite)
        
    dwcm9_value = np.mean(vulnerability_scores)

    return {
        'N3': n3_value,
        'L2': l2_value,
        'dwCM9': dwcm9_value
    }
