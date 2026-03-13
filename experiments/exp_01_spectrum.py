import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings

# Asegurar que Python encuentre nuestra librería src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import compute_complexity_metrics

warnings.filterwarnings('ignore')
print("Iniciando Escáner Topológico de Espectro Completo (Minoritaria vs Mayoritaria)...")

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
# Usamos el path estructurado o un dataset sintético como fallback seguro
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'yeast1.csv'))
try:
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print("Dataset cargado exitosamente.")
except FileNotFoundError:
    print("Dataset no encontrado en data/raw/. Generando hiperespacio sintético de prueba...")
    X, y = make_classification(n_samples=2000, n_features=8, n_informative=4, 
                               weights=[0.95, 0.05], random_state=42)

clases, conteos = np.unique(y, return_counts=True)
clase_min = clases[np.argmin(conteos)]
clase_maj = clases[np.argmax(conteos)]
ratio_original = conteos[np.argmin(conteos)] / conteos[np.argmax(conteos)]

# ==========================================
# 2. CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
ratios_prueba = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
diccionario_samplers = {
    'SMOTE': SMOTE,
    'ADASYN': ADASYN,
    'ROS': RandomOverSampler,
    'RUS': RandomUnderSampler
}

metricas_keys = ['N3', 'L2', 'dwCM9']
resultados_min = {m: {s: [] for s in diccionario_samplers} for m in metricas_keys}
resultados_maj = {m: {s: [] for s in diccionario_samplers} for m in metricas_keys}

# Calcular estado original (Baseline) para AMBAS clases
base_min = compute_complexity_metrics(X, y, target_class=clase_min)
base_maj = compute_complexity_metrics(X, y, target_class=clase_maj)

for m in metricas_keys:
    for s in diccionario_samplers:
        resultados_min[m][s].append(base_min[m])
        resultados_maj[m][s].append(base_maj[m])

eje_x_ratios = [ratio_original] + ratios_prueba

# ==========================================
# 3. BARRIDO HIPERESPACIAL
# ==========================================
print("Inyectando/eliminando datos y registrando perturbaciones espaciales...")
for r in ratios_prueba:
    for nombre, SamplerClass in diccionario_samplers.items():
        try:
            sampler = SamplerClass(sampling_strategy=r, random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
            
            # ¡Llamada doble a su módulo!
            res_min = compute_complexity_metrics(X_res, y_res, target_class=clase_min)
            res_maj = compute_complexity_metrics(X_res, y_res, target_class=clase_maj)
            
            for m in metricas_keys:
                resultados_min[m][nombre].append(res_min[m])
                resultados_maj[m][nombre].append(res_maj[m])
                
        except Exception:
            # Tolerancia a fallos geométricos (ej. falta de vecinos en ADASYN)
            for m in metricas_keys:
                resultados_min[m][nombre].append(resultados_min[m][nombre][-1])
                resultados_maj[m][nombre].append(resultados_maj[m][nombre][-1])

# --- Exportar Tabla a CSV ---
print("Compilando matriz de datos para exportación...")
filas_csv = []
for i, ratio in enumerate(eje_x_ratios):
    for sampler in diccionario_samplers:
        fila = {'Ratio': ratio, 'Sampler': sampler}
        for m in metricas_keys:
            fila[f'Min_{m}'] = resultados_min[m][sampler][i]
            fila[f'Maj_{m}'] = resultados_maj[m][sampler][i]
        filas_csv.append(fila)

df_export = pd.DataFrame(filas_csv)
dir_tables = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'tables'))
os.makedirs(dir_tables, exist_ok=True)
ruta_csv = os.path.join(dir_tables, 'exp_05_topological_spectrum.csv')
df_export.to_csv(ruta_csv, index=False)
print(f" Tabla exportada: {ruta_csv}")

# ==========================================
# 4. VISUALIZACIÓN: DASHBOARD DUAL (2x3)
# ==========================================
print("Renderizando Centro de Comando Visual (2x3)...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(22, 12))

colores = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
marcadores = ['o', 's', '^', 'D']
titulos_metricas = ['N3 (Solapamiento 1-NN)', 'L2 (Error Lineal SVC)', 'dwCM9 (Vulnerabilidad Vecinal)']

for col_idx, met_key in enumerate(metricas_keys):
    # Fila 0: CLASE MINORITARIA
    ax_min = axes[0, col_idx]
    for (nombre, trayectoria), color, marcador in zip(resultados_min[met_key].items(), colores, marcadores):
        ax_min.plot(eje_x_ratios, trayectoria, marker=marcador, markersize=8, linewidth=2.5, label=nombre, color=color, alpha=0.85)
    
    ax_min.axvline(x=ratio_original, color='gray', linestyle='--', linewidth=1.5)
    ax_min.set_title(f'[MINORITARIA] {titulos_metricas[col_idx]}\n(Curvas que bajan = Mejora)', fontsize=13, pad=10, fontweight='bold')
    if col_idx == 0: ax_min.set_ylabel('Valor de Complejidad', fontsize=12)
    ax_min.legend(loc='best')

    # Fila 1: CLASE MAYORITARIA
    ax_maj = axes[1, col_idx]
    for (nombre, trayectoria), color, marcador in zip(resultados_maj[met_key].items(), colores, marcadores):
        ax_maj.plot(eje_x_ratios, trayectoria, marker=marcador, markersize=8, linewidth=2.5, label=nombre, color=color, alpha=0.85)
    
    ax_maj.axvline(x=ratio_original, color='gray', linestyle='--', linewidth=1.5)
    ax_maj.set_title(f'[MAYORITARIA] {titulos_metricas[col_idx]}\n(Curvas que suben = Daño Colateral)', fontsize=13, pad=10, fontweight='bold', color='#c0392b')
    ax_maj.set_xlabel('Ratio de Muestreo Objetivo', fontsize=12)
    if col_idx == 0: ax_maj.set_ylabel('Valor de Complejidad', fontsize=12)

plt.suptitle('El Principio de Conservación Topológica en Datasets Desbalanceados:\nMejora Minoritaria vs. Daño Mayoritario', fontsize=18, fontweight='bold', y=1.04)
plt.tight_layout()



# --- Exportar Gráfico a PNG ---
dir_figures = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures'))
os.makedirs(dir_figures, exist_ok=True)
ruta_png = os.path.join(dir_figures, 'exp_05_topological_spectrum.png')
plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
print(f" Gráfico exportado: {ruta_png}")

plt.show()
