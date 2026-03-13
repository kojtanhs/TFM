import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
import warnings

# Asegurar que Python encuentre nuestra librería src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import hostility_measure, topological_ratio_optimizer

warnings.filterwarnings('ignore')
print("Iniciando Benchmark de Coste Computacional (Complejidad Algorítmica)...")

# ==========================================
# 1. PREPARACIÓN DE ESCENARIOS
# ==========================================
print("Configurando escenarios topológicos...")

# Escenario 1: Alta Complejidad (Requiere sobremuestreo iterativo)
# Intentamos cargar el dataset real, si no, generamos uno de alta dificultad
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'yeast1.csv'))
try:
    df_real = pd.read_csv(data_path)
    X_comp = df_real.iloc[:, :-1].values
    y_comp = df_real.iloc[:, -1].values
except FileNotFoundError:
    X_comp, y_comp = make_classification(n_samples=1500, n_features=8, n_informative=4, 
                                         weights=[0.95, 0.05], class_sep=0.8, random_state=42)

# Escenario 2: Baja Complejidad (Clases separadas, el algoritmo debe abortar rápido)
X_easy, y_easy = make_classification(
    n_samples=3000, n_features=8, n_informative=4, n_redundant=0,
    n_clusters_per_class=1, weights=[0.99, 0.01], class_sep=3.0, random_state=42
)

escenarios = {
    'Escenario 1 (Alta Complejidad / Solapamiento)': (X_comp, y_comp),
    'Escenario 2 (Baja Complejidad / Separación)': (X_easy, y_easy)
}

resultados_tiempos = []

# ==========================================
# 2. EJECUCIÓN DEL BENCHMARK
# ==========================================
for nombre_escenario, (X, y) in escenarios.items():
    print(f"Evaluando: {nombre_escenario}")
    
    # Medir HARS Original (Lancho)
    start_time = time.time()
    try:
        hostility_measure(X, y, sigma=5, delta=0.5, k_min=0, seed=42)
        t_hars = time.time() - start_time
    except Exception:
        t_hars = float('nan')

    # Medir Barella-HARS (Optimizador Topológico)
    start_time = time.time()
    try:
        topological_ratio_optimizer(X, y, theta=0.05, step=0.05)
        t_barella = time.time() - start_time
    except Exception:
        t_barella = float('nan')

    # Calcular Aceleración
    speedup_val = t_hars / t_barella if t_barella > 0 else float('nan')
    
    resultados_tiempos.append({
        'Escenario': nombre_escenario,
        'HARS Original (seg)': t_hars,
        'Barella-HARS (seg)': t_barella,
        'Speedup': speedup_val
    })

# ==========================================
# 3. PROCESAMIENTO Y EXPORTACIÓN DE TABLA
# ==========================================
df_tiempos = pd.DataFrame(resultados_tiempos).set_index('Escenario')

print("\n" + "="*80)
print("ANÁLISIS DE COSTE COMPUTACIONAL: ESTADO DEL ARTE VS. OPTIMIZACIÓN VECTORIAL")
print("="*80)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df_tiempos)
print("="*80 + "\n")

# Guardar CSV
dir_tables = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'tables'))
os.makedirs(dir_tables, exist_ok=True)
ruta_csv = os.path.join(dir_tables, 'exp_10_computational_benchmark.csv')
df_tiempos.to_csv(ruta_csv, index=True)
print(f"Tabla de tiempos exportada a: {ruta_csv}")

# ==========================================
# 4. VISUALIZACIÓN Y EXPORTACIÓN DE GRÁFICO
# ==========================================
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df_tiempos.index))
width = 0.35

color_hars = '#e74c3c'
color_barella = '#2ecc71'

rects1 = ax.bar(x - width/2, df_tiempos['HARS Original (seg)'], width, label='HARS Original (Clustering Jerárquico)', color=color_hars)
rects2 = ax.bar(x + width/2, df_tiempos['Barella-HARS (seg)'], width, label='Barella-HARS (k-NN Vectorizado)', color=color_barella)

ax.set_ylabel('Tiempo de Ejecución (Segundos)', fontsize=12, fontweight='bold')
ax.set_title('Comparativa de Eficiencia Algorítmica: Coste Computacional por Escenario', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([idx.replace(' (', '\n(') for idx in df_tiempos.index], fontsize=11)
ax.legend(fontsize=11)

# Añadir etiquetas de texto sobre las barras
def autolabel(rects, is_barella=False, speedup_data=None):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        texto = f'{height:.2f}s'
        if is_barella and speedup_data is not None:
            texto += f'\n({speedup_data[i]:.2f}x Speedup)'
            
        ax.annotate(texto,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2, is_barella=True, speedup_data=df_tiempos['Speedup'].values)



plt.tight_layout()

# Guardar Gráfico
dir_figures = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures'))
os.makedirs(dir_figures, exist_ok=True)
ruta_png = os.path.join(dir_figures, 'exp_10_computational_benchmark.png')
plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
print(f"Gráfico de coste computacional exportado a: {ruta_png}")

plt.show()
