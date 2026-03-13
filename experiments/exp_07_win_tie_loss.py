import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.metrics import geometric_mean_score
import warnings

# Asegurar que Python encuentre nuestra librería src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import topological_ratio_optimizer

warnings.filterwarnings('ignore')
print("Iniciando Análisis Win-Tie-Loss Multi-Clasificador (Validación de Robustez)...")

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'yeast1.csv'))
try:
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
except FileNotFoundError:
    print("Dataset no encontrado. Verifique la carpeta data/raw/")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# ==========================================
# 2. OBTENER EL RATIO ÓPTIMO TOPOLÓGICO
# ==========================================
print("Calculando ratio óptimo mediante Barella-HARS...")
ratio_optimo, _ = topological_ratio_optimizer(X_train, y_train, theta=0.05, step=0.05)
print(f"Ratio de equilibrio topológico: {ratio_optimo:.3f}")

# ==========================================
# 3. CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
clasificadores = {
    'RF': RandomForestClassifier(random_state=42, n_jobs=-1),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(random_state=42),
    'MLP': MLPClassifier(random_state=42, max_iter=500),
    'GB': HistGradientBoostingClassifier(random_state=42)
}

samplers = {
    'ROS': RandomOverSampler,
    'SMOTE': SMOTE,
    'ADASYN': ADASYN,
    'SMOTE+Tomek': SMOTETomek
}

UMBRAL_EMPATE = 0.01 # Diferencia de G-mean menor al 1% se considera empate
resultados_wtl = {nombre: {'Win': 0, 'Tie': 0, 'Loss': 0} for nombre in samplers}

# ==========================================
# 4. BUCLE DE EVALUACIÓN CRUZADA
# ==========================================
print("Evaluando combinaciones Clasificador-Sampler...")
for nombre_sampler, ClaseSampler in samplers.items():
    for nombre_clf, clf in clasificadores.items():
        
        # --- A. Evaluar Balanceo Tradicional (Ratio 1.0) ---
        try:
            if nombre_sampler == 'SMOTE+Tomek':
                smote_bal = SMOTE(sampling_strategy=1.0, random_state=42)
                sampler_bal = ClaseSampler(smote=smote_bal, random_state=42)
            else:
                sampler_bal = ClaseSampler(sampling_strategy=1.0, random_state=42)
            
            X_bal, y_bal = sampler_bal.fit_resample(X_train, y_train)
            clf.fit(X_bal, y_bal)
            gmean_bal = geometric_mean_score(y_test, clf.predict(X_test))
        except Exception:
            gmean_bal = 0
            
        # --- B. Evaluar Propuesta Barella-HARS (Ratio Óptimo) ---
        try:
            if nombre_sampler == 'SMOTE+Tomek':
                smote_opt = SMOTE(sampling_strategy=ratio_optimo, random_state=42)
                sampler_opt = ClaseSampler(smote=smote_opt, random_state=42)
            else:
                sampler_opt = ClaseSampler(sampling_strategy=ratio_optimo, random_state=42)
            
            X_opt, y_opt = sampler_opt.fit_resample(X_train, y_train)
            clf.fit(X_opt, y_opt)
            gmean_opt = geometric_mean_score(y_test, clf.predict(X_test))
        except Exception:
            gmean_opt = 0

        # Lógica Win-Tie-Loss
        diferencia = gmean_opt - gmean_bal
        if abs(diferencia) <= UMBRAL_EMPATE:
            resultados_wtl[nombre_sampler]['Tie'] += 1
        elif diferencia > UMBRAL_EMPATE:
            resultados_wtl[nombre_sampler]['Win'] += 1
        else:
            resultados_wtl[nombre_sampler]['Loss'] += 1

# ==========================================
# 5. VISUALIZACIÓN Y EXPORTACIÓN
# ==========================================
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 7))

labels = list(resultados_wtl.keys())
wins = [resultados_wtl[l]['Win'] for l in labels]
ties = [resultados_wtl[l]['Tie'] for l in labels]
losses = [resultados_wtl[l]['Loss'] for l in labels]

x = np.arange(len(labels))
width = 0.6

# Colores estándar para Win-Tie-Loss
color_win = '#2ecc71'
color_tie = '#95a5a6'
color_loss = '#e74c3c'

p1 = ax.bar(x, wins, width, label='Victoria (Barella-HARS > 1.0)', color=color_win)
p2 = ax.bar(x, ties, width, bottom=wins, label='Empate (Diferencia < 1%)', color=color_tie)
p3 = ax.bar(x, losses, width, bottom=np.array(wins)+np.array(ties), label='Derrota (Ratio 1.0 > Barella)', color=color_loss)

ax.set_ylabel('Conteo de Clasificadores', fontsize=12, fontweight='bold')
ax.set_title('Análisis Win-Tie-Loss: Robustez del Equilibrio Topológico vs. Balanceo Total', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Resultado")

for c in ax.containers:
    ax.bar_label(c, label_type='center', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()

# Exportar Gráfico
dir_figures = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures'))
os.makedirs(dir_figures, exist_ok=True)
ruta_png = os.path.join(dir_figures, 'exp_11_win_tie_loss_master.png')
plt.savefig(ruta_png, dpi=300, bbox_inches='tight')

# Exportar Tabla
df_wtl = pd.DataFrame(resultados_wtl).T
dir_tables = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'tables'))
os.makedirs(dir_tables, exist_ok=True)
ruta_csv = os.path.join(dir_tables, 'exp_11_win_tie_loss_results.csv')
df_wtl.to_csv(ruta_csv)

print(f" Análisis completado. Gráfico: {ruta_png}, Tabla: {ruta_csv}")
plt.show()
