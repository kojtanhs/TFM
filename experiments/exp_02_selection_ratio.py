import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score
import warnings

# Asegurar que Python encuentre nuestra librería src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import compute_complexity_metrics

warnings.filterwarnings('ignore')
print("Iniciando Búsqueda del Ratio Óptimo (Topología vs. Rendimiento Predictivo)...")

# ==========================================
# 1. CARGA Y DIVISIÓN DE DATOS
# ==========================================
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'yeast1.csv'))
try:
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
except FileNotFoundError:
    print("Dataset no encontrado. Asegúrese de que existe en data/raw/")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

clases, conteos = np.unique(y_train, return_counts=True)
clase_min = clases[np.argmin(conteos)]
ratio_original = conteos[np.argmin(conteos)] / conteos[np.argmax(conteos)]

# ==========================================
# 2. VARIABLES Y BASELINE
# ==========================================
ratios_prueba = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
resultados = []

print("Evaluando estado original (Baseline)...")
metricas_base = compute_complexity_metrics(X_train, y_train, target_class=clase_min)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

resultados.append({
    'Ratio': round(ratio_original, 3),
    'N3 (Solapamiento)': metricas_base['N3'],
    'dwCM9 (Vulnerabilidad)': metricas_base['dwCM9'],
    'G-mean': geometric_mean_score(y_test, y_pred_base),
    'F1-score': f1_score(y_test, y_pred_base, pos_label=clase_min)
})

# ==========================================
# 3. BARRIDO DE MUESTREO Y EVALUACIÓN
# ==========================================
print("Inyectando datos con SMOTE y evaluando rendimiento...")
for r in ratios_prueba:
    smote = SMOTE(sampling_strategy=r, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Auditoría Topológica
    metricas = compute_complexity_metrics(X_res, y_res, target_class=clase_min)
    
    # Entrenamiento y Evaluación
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_res, y_res)
    y_pred = rf.predict(X_test)
    
    resultados.append({
        'Ratio': r,
        'N3 (Solapamiento)': metricas['N3'],
        'dwCM9 (Vulnerabilidad)': metricas['dwCM9'],
        'G-mean': geometric_mean_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred, pos_label=clase_min)
    })

# ==========================================
# 4. PROCESAMIENTO Y EXPORTACIÓN DE TABLA
# ==========================================
df_res = pd.DataFrame(resultados).set_index('Ratio')

ratio_optimo = df_res['G-mean'].idxmax()
mejor_gmean = df_res.loc[ratio_optimo, 'G-mean']
mejor_f1 = df_res.loc[ratio_optimo, 'F1-score']

print(f"\n✅ BÚSQUEDA COMPLETADA:")
print(f"🏆 Ratio Óptimo Encontrado: {ratio_optimo}")
print(f"   - G-mean Máximo: {mejor_gmean:.4f}")
print(f"   - F1-score: {mejor_f1:.4f}")
print(f"⚠️ Rendimiento en Balanceo Total (Ratio 1.0): G-mean {df_res.loc[1.0, 'G-mean']:.4f}\n")

dir_tables = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'tables'))
os.makedirs(dir_tables, exist_ok=True)
ruta_csv = os.path.join(dir_tables, 'exp_06_optimal_ratio_search.csv')
df_res.to_csv(ruta_csv, index=True) # Mantenemos el índice porque es el Ratio
print(f"✅ Tabla exportada exitosamente a: {ruta_csv}")

# ==========================================
# 5. VISUALIZACIÓN Y EXPORTACIÓN DE GRÁFICO
# ==========================================
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(12, 7))

# Eje Izquierdo: Rendimiento Predictivo
color_gmean = '#8e44ad'
color_f1 = '#2980b9'

linea1 = ax1.plot(df_res.index, df_res['G-mean'], marker='D', markersize=8, linewidth=3, color=color_gmean, label='G-mean (Test)')
linea2 = ax1.plot(df_res.index, df_res['F1-score'], marker='s', markersize=8, linewidth=3, color=color_f1, label='F1-score (Test)')

ax1.set_xlabel('Ratio de Muestreo Objetivo (SMOTE)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Rendimiento del Clasificador (Mayor = Mejor)', fontsize=13, fontweight='bold')
ax1.tick_params(axis='y')

# Eje Derecho: Complejidad Topológica
ax2 = ax1.twinx()
color_n3 = '#27ae60'
color_dwcm = '#e67e22'

linea3 = ax2.plot(df_res.index, df_res['N3 (Solapamiento)'], marker='o', markersize=6, linewidth=2, linestyle='--', color=color_n3, label='N3 (Solapamiento Local)')
linea4 = ax2.plot(df_res.index, df_res['dwCM9 (Vulnerabilidad)'], marker='^', markersize=6, linewidth=2, linestyle='--', color=color_dwcm, label='dwCM9 (Vulnerabilidad Topológica)')

ax2.set_ylabel('Métricas de Complejidad (Menor = Mejor)', fontsize=13, fontweight='bold')
ax2.tick_params(axis='y')

# Marcar el Ratio Óptimo
ax1.axvline(x=ratio_optimo, color='gold', linestyle='-', linewidth=2.5, zorder=0)
ax1.text(ratio_optimo + 0.01, ax1.get_ylim()[0] + 0.02, f'Ratio Óptimo\n({ratio_optimo})', color='goldenrod', fontweight='bold', fontsize=11)

# Unificar leyendas
lineas = linea1 + linea2 + linea3 + linea4
etiquetas = [l.get_label() for l in lineas]
ax1.legend(lineas, etiquetas, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=11)



plt.title('Selección del Ratio Óptimo: Intersección entre Topología y Generalización', fontsize=16, pad=20, fontweight='bold')
plt.tight_layout()

dir_figures = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures'))
os.makedirs(dir_figures, exist_ok=True)
ruta_png = os.path.join(dir_figures, 'exp_06_optimal_ratio_search.png')
plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico exportado exitosamente a: {ruta_png}")

plt.show()
