# Predicción de Enfermedades del Corazón – Regresión Logística

## Introducción

Este proyecto tiene como objetivo implementar un modelo de **Regresión Logística** para predecir la presencia de enfermedades del corazón utilizando datos clínicos. La idea es trabajar paso a paso, desde la exploración inicial de los datos hasta el despliegue del modelo en un entorno de producción (Amazon SageMaker).

Me enfoqué en:
- Que el preprocesamiento y el entrenamiento del modelo sean claros y reproducibles.
- Que las métricas y visualizaciones cuenten una historia coherente sobre el desempeño del modelo.

El dataset contiene 270 registros de pacientes con 14 características clínicas, como edad, colesterol, presión arterial, entre otras. La variable objetivo es binaria (`1` para presencia y `0` para ausencia de enfermedad).

---

## Qué hay en el notebook

### **01_heart_disease_lr_analysis.ipynb**
- **Carga y preparación del dataset**:
  - Exploración inicial (EDA): estadísticas descriptivas, visualización de la distribución de clases.
  - Preprocesamiento: binarización de la variable objetivo, normalización de características y división en entrenamiento/prueba.
- **Implementación de regresión logística** *(en progreso)*:
  - Funciones básicas: sigmoide, cálculo del costo (entropía cruzada) y gradiente descendente.
  - Entrenamiento del modelo y evaluación de métricas.
- **Visualización de fronteras de decisión** *(pendiente)*:
  - Entrenamiento en pares de características y análisis de separabilidad.
- **Regularización L2** *(pendiente)*:
  - Comparación de métricas con y sin regularización.
- **Despliegue en SageMaker** *(pendiente)*:
  - Exportación del modelo y creación de un endpoint para inferencias en tiempo real.

---

## Requisitos

- Python 3.x
- Bibliotecas necesarias:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Instala las dependencias ejecutando:
```bash
pip install -r requirements.txt
```
## Información del Proyecto

- **Autor**: Esteban Aguilera Contreras
- **Universidad**: Escuela Colombiana de Ingeniería Julio Garavito
- **Asignatura**: Arquitecturas Empresariales (AREP)
