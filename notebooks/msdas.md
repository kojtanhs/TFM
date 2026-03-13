# Interpretación de Resultados: Optimización Topológica en Aprendizaje Desbalanceado

El presente documento expone las conclusiones extraídas de la experimentación computacional realizada sobre espacios de características desbalanceados. A través de la implementación de tres cuadernos de análisis interactivo (Jupyter Notebooks), se ha evaluado la viabilidad de sustituir el balanceo de clases tradicional (ratio 1.0) por un enfoque basado en la complejidad geométrica y la vulnerabilidad topológica.

Los resultados aquí expuestos sientan las bases de una primera propuesta metodológica firme para la comunidad científica, demostrando que la optimización del hiperespacio es superior a la igualación volumétrica de las clases.

---

## 1. Análisis Topológico Visual (Notebook 01)
**El Principio de Mínima Modificación Geométrica**

El primer experimento visualiza el impacto directo de la inyección de datos sintéticos sobre la distribución bidimensional de las clases. El objetivo fue contrastar el estado original del conjunto de datos frente a dos estrategias de sobremuestreo: el balanceo total y el punto óptimo topológico (calculado iterativamente).

* **La falacia del balanceo total:** Se ha evidenciado gráficamente que forzar un ratio de 1.0 genera estructuras macizas y artificiales. Esta saturación del hiperespacio destruye las fronteras naturales de la clase mayoritaria, provocando un solapamiento masivo que, teóricamente, induce al sobreajuste (overfitting).
* **El equilibrio topológico:** El algoritmo propuesto logra densificar los subgrupos aislados de la clase minoritaria inyectando únicamente la cantidad de vectores sintéticos necesarios para reducir su vulnerabilidad espacial. 



**Conclusión Preliminar I:** Modificar la distribución de los datos lo mínimo indispensable para igualar las complejidades inter-clase protege la capacidad de generalización del espacio original.

---

## 2. Dinámica Evolutiva de las Métricas (Notebook 02)
**Funciones Discretas vs. Gradientes Continuos**

El segundo análisis profundiza en la naturaleza matemática de las funciones de coste utilizadas para determinar el umbral de parada temprana durante el sobremuestreo, comparando la medida de Hostilidad original del estado del arte frente a la métrica de vulnerabilidad geométrica $dwCM_9$.

* **Limitaciones de la métrica de Hostilidad:** Al fundamentarse en un umbral estricto (función escalón), la métrica de Hostilidad original se muestra insensible a los cambios sutiles de densidad generados iterativamente por algoritmos como SMOTE o ADASYN. Las curvas de evolución tienden a ser planas y escalonadas, requiriendo una expansión forzada del vecindario ($k$-NN) para percibir alteraciones.
* **La superioridad analítica de $dwCM_9$:** La métrica basada en la formulación de Barella proporciona un gradiente continuo y de alta resolución. Esto permite trazar curvas suaves que reflejan exactamente cómo la clase minoritaria gana densidad al mismo tiempo que la mayoritaria pierde integridad.



**Conclusión Preliminar II:** Para optimizar dinámicamente el sobremuestreo en conjuntos de datos complejos, es imperativo abandonar funciones de coste basadas en umbrales rígidos en favor de métricas continuas que ponderen la densidad espacial inversa, garantizando así un cálculo preciso de la intersección topológica.

---

## 3. Barrido de Rendimiento Predictivo (Notebook 03)
**Traducción del Equilibrio Geométrico a la Capacidad de Generalización**

El tercer experimento cierra la brecha entre la teoría topológica y el modelado predictivo aplicado. Se realizó un barrido paramétrico integrando la auditoría espacial (medición de $N_3$, $L_2$ y $dwCM_9$) con el rendimiento de un clasificador subyacente medido a través de la métrica G-mean.

* **El punto de máxima generalización:** Los resultados empíricos confirman que el rendimiento predictivo del clasificador se maximiza de forma congruente en las iteraciones previas al balanceo absoluto. 
* **Correlación directa:** Se observa una correlación verificable entre el descenso sostenido de la métrica $dwCM_9$ (reducción de vulnerabilidad) y el aumento del G-mean. Sin embargo, cuando el ratio se aproxima a 1.0, las métricas de daño a la clase mayoritaria se disparan, lo que se traduce en una caída empírica del rendimiento en el conjunto de prueba.

**Conclusión Preliminar III:** El rendimiento predictivo óptimo en problemas de clasificación desbalanceada no se encuentra en la paridad de clases, sino en el punto matemático exacto donde la complejidad topológica de ambas clases se interseca.

---

## Consideraciones Finales y Trabajo Futuro

Las evidencias recolectadas en estos tres escenarios proponen un cambio de paradigma en el preprocesamiento de datos: la transición del **balanceo volumétrico** al **balanceo topológico**. Si bien los resultados exhiben una robustez matemática y computacional notable, se reconoce que esta propuesta constituye una primera aproximación al problema.

Para consolidar este marco teórico, las futuras líneas de investigación deberán someter esta arquitectura a validaciones más extensas, incluyendo análisis sobre conjuntos de datos de dimensionalidad extrema y evaluaciones frente a ensambles de clasificadores más heterogéneos. No obstante, la arquitectura actual demuestra que operar bajo el principio de "mínima modificación geométrica" es un enfoque superior, eficiente y metodológicamente más riguroso para la ciencia de datos contemporánea.
