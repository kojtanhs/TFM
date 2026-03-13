import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import warnings

# Asegurar que Python encuentre nuestra librería src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import hostility_measure, compute_complexity_metrics

warnings.filterwarnings('ignore')
print("Iniciando Demostración Visual 2D (Réplica de Sección 4.1)...")

# ==========================================
# 1. CREACIÓN DEL ESPACIO TOPOLÓGICO SINTÉTICO
# ==========================================
# Diseñamos un escenario con alto desbalance (10% minoritaria, 90% mayoritaria) y solapamiento
X_art, y_art = make_classification(
    n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0.05, 
    class_sep=0.8, random_state=42
)

clase_min = 1
clase_maj = 0
n_min = np.sum(y_art == clase_min)
n_maj = np.sum(y_art == clase_maj)

# ==========================================
# 2. CÁLCULO DE RATIOS (HARS vs BALANCEO TOTAL)
# ==========================================
ratio_original = n_min / n_maj
ratio_smote = 1.0

print("Calculando medida de Hostilidad sobre el espacio artificial...")
_, _, _, k_auto = hostility_measure(X_art, y_art, sigma=5, delta=0.5, k_min=0, seed=42)
ratio_hars = (n_min + k_auto) / n_maj

# ==========================================
# 3. APLICACIÓN DE MUESTREO SINTÉTICO
# ==========================================
print("Inyectando instancias sintéticas...")
smote_total = SMOTE(sampling_strategy=ratio_smote, random_state=42)
X_total, y_total = smote_total.fit_resample(X_art, y_art)

smote_hars = SMOTE(sampling_strategy=ratio_hars, random_state=42)
X_hars, y_hars = smote_hars.fit_resample(X_art, y_art)

# ==========================================
# 4. AUDITORÍA TOPOLÓGICA (COMPLEJIDAD GEOMÉTRICA)
# ==========================================
print("Evaluando la vulnerabilidad resultante...")
met_orig = compute_complexity_metrics(X_art, y_art, target_class=clase_min)
met_hars = compute_complexity_metrics(X_hars, y_hars, target_class=clase_min)
met_total = compute_complexity_metrics(X_total, y_total, target_class=clase_min)

# ==========================================
# 5. VISUALIZACIÓN Y EXPORTACIÓN
# ==========================================
print("Renderizando comparativa espacial 2D...")
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

titulos = [
    f'A) Original (Ratio {ratio_original:.2f})\nN3: {met_orig["N3"]:.3f} | dwCM9: {met_orig["dwCM9"]:.3f}',
    f'B) HARS Óptimo (Ratio {ratio_hars:.2f})\nN3: {met_hars["N3"]:.3f} | dwCM9: {met_hars["dwCM9"]:.3f}',
    f'C) SMOTE Total (Ratio {ratio_smote:.2f})\nN3: {met_total["N3"]:.3f} | dwCM9: {met_total["dwCM9"]:.3f}'
]

datasets = [(X_art, y_art), (X_hars, y_hars), (X_total, y_total)]
color_maj = '#bdc3c7' # Gris claro para que la mayoritaria quede en segundo plano
color_min = '#c0392b' # Rojo oscuro para destacar la estructura minoritaria

for i, ax in enumerate(axes):
    X_plot, y_plot = datasets[i]
    
    # Mayoritaria al fondo
    ax.scatter(X_plot[y_plot == clase_maj][:, 0], X_plot[y_plot == clase_maj][:, 1], 
               c=color_maj, alpha=0.5, s=30, label='Clase Mayoritaria' if i==0 else "")
    
    # Minoritaria al frente
    ax.scatter(X_plot[y_plot == clase_min][:, 0], X_plot[y_plot == clase_min][:, 1], 
               c=color_min, edgecolor='white', linewidth=0.5, s=45, label='Clase Minoritaria' if i==0 else "")
    
    ax.set_title(titulos[i], fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Dimensión 1', fontsize=12)
    if i == 0:
        ax.set_ylabel('Dimensión 2', fontsize=12)
        ax.legend(loc='upper right', frameon=True)

plt.suptitle('Demostración Visual del Muestreo Sintético: Mínima Modificación vs. Sobre-Generalización', fontsize=16, y=1.05, fontweight='bold')
plt.tight_layout()



# Exportar Gráfico a PNG
dir_figures = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures'))
os.makedirs(dir_figures, exist_ok=True)
ruta_png = os.path.join(dir_figures, 'exp_08_2d_visual_hars.png')
plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
print(f" Gráfico exportado exitosamente a: {ruta_png}")

plt.show()
