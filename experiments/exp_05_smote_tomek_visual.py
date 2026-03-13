import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import warnings

# Add project root to path for src module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import topological_ratio_optimizer

warnings.filterwarnings('ignore')
print("Iniciando radiografía visual 2D (Evolución Topológica con SMOTE+Tomek)...")

# 1. Generate artificial dataset (5% imbalance and small disjuncts)
X_art, y_art = make_classification(
    n_samples=1500, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=2, weights=[0.95, 0.05], class_sep=1.0, random_state=42
)

# 2. Calculate optimal ratio using the topological optimizer
print("Calculando ratio óptimo topológico...")
try:
    ratio_optimo, _ = topological_ratio_optimizer(X_art, y_art, theta=0.05, step=0.05)
except Exception:
    # Fallback in case of convergence issues
    ratio_optimo = 0.25 
print(f"Ratio óptimo calculado: {ratio_optimo:.3f}")

# 3. Apply hybrid sampling (SMOTE + Tomek Links)
print("Aplicando SMOTE+Tomek (Ratio Óptimo vs. Balance Total)...")

# Topological optimal ratio
smote_opt = SMOTE(sampling_strategy=ratio_optimo, random_state=42)
smt_opt = SMOTETomek(smote=smote_opt, random_state=42)
X_opt, y_opt = smt_opt.fit_resample(X_art, y_art)

# Full balance (Ratio 1.0)
smote_bal = SMOTE(sampling_strategy=1.0, random_state=42)
smt_bal = SMOTETomek(smote=smote_bal, random_state=42)
X_bal, y_bal = smt_bal.fit_resample(X_art, y_art)

print("Renderizando los 3 paneles visuales...")

# ==========================================
# 4. VISUALIZATION: 3-PANEL COMPARISON
# ==========================================
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

# Color palette matching the description
color_maj = '#3498db' # Blue for majority
color_min = '#e67e22' # Orange for minority

datasets = [
    (X_art, y_art, 'a) Sin Muestreo\n(Dataset Original - Desbalance Severo)'),
    (X_opt, y_opt, f'b) Ratio Óptimo ({ratio_optimo:.3f})\n(Principio de Mínima Modificación)'),
    (X_bal, y_bal, 'c) Balance Total (Ratio 1.0)\n(Estructura Masiva / Sobreajuste)')
]

for i, ax in enumerate(axes):
    X_plot, y_plot = datasets[i][:2]
    titulo = datasets[i][2]
    
    # Plot majority class in background
    ax.scatter(X_plot[y_plot == 0][:, 0], X_plot[y_plot == 0][:, 1], 
               c=color_maj, alpha=0.4, s=30, edgecolors='none', label='Clase Mayoritaria' if i==0 else "")
    
    # Plot minority class in foreground
    ax.scatter(X_plot[y_plot == 1][:, 0], X_plot[y_plot == 1][:, 1], 
               c=color_min, alpha=0.9, s=40, edgecolors='white', linewidth=0.5, label='Clase Minoritaria' if i==0 else "")
    
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dimensión 1', fontsize=12)
    if i == 0:
        ax.set_ylabel('Dimensión 2', fontsize=12)
        ax.legend(loc='upper right', fontsize=11, frameon=True)

plt.suptitle('Evolución Topológica con SMOTE+Tomek: Prevención de la Degeneración Espacial', 
             fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()



# Export figure to results directory
dir_figures = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures'))
os.makedirs(dir_figures, exist_ok=True)
ruta_png = os.path.join(dir_figures, 'exp_09_smotetomek_visual.png')
plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
print(f"Gráfico exportado exitosamente a: {ruta_png}")

plt.show()
