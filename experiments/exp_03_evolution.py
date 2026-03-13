import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings

# Asegurar que Python encuentre nuestra librería src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import compute_complexity_metrics

warnings.filterwarnings('ignore')
print("Iniciando Análisis de Intersección Topológica (Equilibrio Geométrico SMOTE)...")

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'yeast1.csv'))
try:
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
except FileNotFoundError:
    print("Dataset no encontrado en data/raw/. Asegúrese de tener el archivo correcto.")
    sys.exit(1)

clases, conteos = np.unique(y, return_counts=True)
clase_min = clases[np.argmin(conteos)]
clase_maj = clases[np.argmax(conteos)]
ratio_original = conteos[np.argmin(conteos)] / conteos[np.argmax(conteos)]

# ==========================================
# 2. CONFIGURACIÓN DEL BARRIDO
# ==========================================
ratios_prueba = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
eje_x_ratios = [ratio_original] + ratios_prueba

metricas_keys = ['N3', 'L2', 'dwCM9']

# Estructuras para almacenar los resultados
resultados_min = {m: [] for m in metricas_keys}
resultados_maj = {m: [] for m in metricas_keys}

# Calcular el estado original (Baseline)
base_min = compute_complexity_metrics(X, y, target_class=clase_min)
base_maj = compute_complexity_metrics(X, y, target_class=clase_maj)

for m in metricas_keys:
    resultados_min[m].append(base_min[m])
    resultados_maj[m].append(base_maj[m])

# ==========================================
# 3. BARRIDO APLICANDO SMOTE
# ==========================================
print("Calculando evolución de métricas para ambas clases...")
for r in ratios_prueba:
    smote = SMOTE(sampling_strategy=r, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Calcular métricas para la clase minoritaria y mayoritaria
    res_min = compute_complexity_metrics(X_res, y_res, target_class=clase_min)
    res_maj = compute_complexity_metrics(X_res, y_res, target_class=clase_maj)
    
    for m in metricas_keys:
        resultados_min[m].append(res_min[m])
        resultados_maj[m].append(res_maj[m])

# ==========================================
# 4. EXPORTACIÓN DE TABLA DE RESULTADOS
# ==========================================
df_resultados = pd.DataFrame({'Ratio_SMOTE': eje_x_ratios})
for m in metricas_keys:
    df_resultados[f'{m}_Minoritaria'] = resultados_min[m]
    df_resultados[f'{m}_Mayoritaria'] = resultados_maj[m]

dir_tables = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'tables'))
os.makedirs(dir_tables, exist_ok=True)
ruta_csv = os.path.join(dir_tables, 'exp_07_topological_equilibrium.csv')
df_resultados.to_csv(ruta_csv, index=False)
print(f" Matriz de intersección exportada exitosamente a: {ruta_csv}")

# ==========================================
# 5. VISUALIZACIÓN: EL CRUCE TOPOLÓGICO
# ==========================================
print("Generando visualización del equilibrio topológico...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

titulos = {
    'N3': 'N3: Solapamiento Local (1-NN)',
    'L2': 'L2: Error Lineal SVC',
    'dwCM9': 'dwCM9: Vulnerabilidad Vecinal'
}

color_min = '#2ecc71' # Verde (Minoritaria)
color_maj = '#e74c3c' # Rojo (Mayoritaria)

for ax, m in zip(axes, metricas_keys):
    # Trazar clase minoritaria
    ax.plot(eje_x_ratios, resultados_min[m], marker='o', markersize=8, linewidth=2.5, 
            color=color_min, label='Clase Minoritaria (C-)')
    
    # Trazar clase mayoritaria
    ax.plot(eje_x_ratios, resultados_maj[m], marker='s', markersize=8, linewidth=2.5, 
            color=color_maj, label='Clase Mayoritaria (C+)')
    
    # Buscar el punto aproximado de cruce para trazar la línea vertical
    y_min_arr = np.array(resultados_min[m])
    y_maj_arr = np.array(resultados_maj[m])
    
    # Lógica para encontrar dónde se cruzan (minoritaria pasa a ser menor o igual que mayoritaria)
    cruce_encontrado = False
    for i in range(1, len(eje_x_ratios)):
        if y_min_arr[i] <= y_maj_arr[i] and y_min_arr[i-1] >= y_maj_arr[i-1]:
            ax.axvline(x=eje_x_ratios[i], color='gray', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(eje_x_ratios[i] - 0.05, ax.get_ylim()[0] + 0.01, 'Cruce\nTopológico', color='#34495e', fontweight='bold', fontsize=10)
            cruce_encontrado = True
            break
            
    # Formato del gráfico
    ax.set_title(titulos[m], fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Ratio Objetivo (SMOTE)', fontsize=12, fontweight='bold')
    if m == 'N3':
        ax.set_ylabel('Valor de Complejidad Geométrica', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    
    # Marcar el ratio original
    ax.axvline(x=ratio_original, color='black', linestyle=':', linewidth=1.5)
    ax.text(ratio_original + 0.01, ax.get_ylim()[1] * 0.95, f'Orig.\n({ratio_original:.2f})', fontsize=10, fontweight='bold')

plt.suptitle('Búsqueda del Equilibrio Geométrico: Intersección de Complejidad inter-clases', fontsize=16, y=1.05, fontweight='bold')
plt.tight_layout()

dir_figures = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures'))
os.makedirs(dir_figures, exist_ok=True)
ruta_png = os.path.join(dir_figures, 'exp_07_topological_equilibrium.png')
plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
print(f" Gráfico de intersección exportado a: {ruta_png}")

plt.show()
