# TFM

La arquitectura de este proyecto se fundamenta rigurosamente en la metodología **HARS** y las directrices establecidas por **Lancho et al. (2025)**.

## Marco Teórico y Propuesta

La investigación parte de la premisa de que el sobremuestreo debe regirse por la topología del dato y no por la paridad aritmética. Para ello, se han integrado y contrastado dos pilares fundamentales de la literatura reciente:

1.  **Metodología HARS (Lancho, 2025)**: Se adopta este marco como base para la determinación del ratio de muestreo óptimo, utilizando la medida de hostilidad como referencia del estado del arte.
2.  **Métricas de Barella (2021)**: Se ponen a disposición las métricas de complejidad geométrica propuestas por Barella para realizar comparaciones empíricas frente a los resultados de Lancho.

El fin último de este trabajo es **extender y completar** las conclusiones de Lancho (2025). Para este propósito, se han desarrollado algoritmos de optimización que toman como punto de partida y evolucionan la lógica de `Hostility_measure_algorithm.py`, buscando una mayor resolución en la detección de la vulnerabilidad espacial.

## Notas de Implementación y Ajustes Técnicos

Para garantizar la robustez del software en entornos de computación científica, se han realizado ajustes críticos en la implementación original de `HARS_algorithm.py`:

* **Estabilidad de Tipos**: Se han refactorizado las asignaciones de variables de control, sustituyendo valores enteros por flotantes (e.g., de `0` a `0.0`) para prevenir errores de casting y asegurar la precisión en cálculos de distancia y densidad.
* **Optimización de Acceso a Datos**: Se ha migrado el uso de indexación basada en `iloc` de Pandas hacia objetos de NumPy y estructuras de arrays nativas. Este cambio reduce la sobrecarga computacional y mejora la compatibilidad con las funciones de búsqueda de vecindad vectorizadas.

## Arquitectura del Repositorio

### 1. Núcleo Algorítmico (`src/`)
* `Hostility_measure_algorithm.py`: Implementación de la medida de hostilidad basada en Lancho (2025).
* `Complexity_metrics_algorithm.py`: Adaptación de las métricas de Barella (2021), incluyendo $N_3, L_2$ y la métrica de vulnerabilidad ponderada $dwCM_9$.
* `Barella_HARS_algorithm.py`: Algoritmo propuesto que extiende la lógica HARS integrando las métricas de complejidad para la búsqueda del ratio óptimo.

### 2. Suite de Experimentos (`experiments/`)
* `exp_01_spectrum.py`: Análisis del espectro topológico bajo diferentes estrategias.
* `exp_02_selection_ratio.py`: Protocolo de selección de ratio basado en la intersección de métricas.
* `exp_03_evolution.py`: Rastreo de la evolución de la complejidad.
* `exp_04_2d_demo.py`: Demostración visual de la metodología en espacios de baja dimensionalidad.
* `exp_05_smote_tomek_visual.py`: Impacto de métodos híbridos en la topología.
* `exp_06_benchmark.py`: Análisis de coste computacional y eficiencia algorítmica.
* `exp_07_win_tie_loss.py`: Validación de robustez mediante análisis multi-clasificador.

### 3. Análisis Interactivo (`notebooks/`)
* `01_EDA_Imbalanced_Data.ipynb`: Análisis topológico inicial.
* `dwCM9_topology.ipynb`: Evaluación profunda de la vulnerabilidad propuesta.
* `Evolution_Metrics_Ratios.ipynb`: Visualización de las curvas de equilibrio.
* `topological_Performance.ipynb`: Correlación entre topología y capacidad predictiva.

## Gestión de Resultados

El sistema centraliza las evidencias en la carpeta `results/`:
* **`tables/`**: Matrices de datos en formato CSV con métricas de rendimiento y complejidad.
* **`figures/`**: Visualizaciones de alta resolución (PNG) que documentan el equilibrio topológico y el análisis Win-Tie-Loss.

---
*Este proyecto se presenta como una propuesta para completar y extender los hallazgos en la optimización del preprocesamiento de datos desbalanceados, estableciendo un puente entre la hostilidad vecinal y la complejidad geométrica.*
