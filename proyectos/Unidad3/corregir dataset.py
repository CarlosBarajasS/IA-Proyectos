import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 1. Cargar tu archivo original (solo parte Gen Z)
df_original = pd.read_csv('dataset_sintetico_5000_ampliado.csv')

# --- PASO 1: ADAPTAR LA PARTE DE GENERACIÓN Z ---
# Renombrar y crear columnas para que coincidan con lo que pide el profe
df_genz = df_original.copy()
df_genz['Categoria'] = 'Generacion Z'

# Asignar Medios aleatorios (mezcla de prensa y redes para Gen Z)
medios_posibles = ['El País', 'El Financiero', 'Reforma', 'El Universal', 'Twitter', 'YouTube', 'TikTok']
# Usamos numpy para asignar pesos (más redes sociales para Gen Z)
pesos_genz = [0.05, 0.05, 0.05, 0.05, 0.3, 0.25, 0.25]
df_genz['Medio'] = np.random.choice(medios_posibles, size=len(df_genz), p=pesos_genz)

# Mapear columnas de texto
df_genz['Titulo'] = "Opinión sobre la crisis generacional" # Título genérico
df_genz['Resumen'] = df_genz['texto'].apply(lambda x: x[:50] + "...") # Primeros 50 caracteres
df_genz['ComentarioReaccion'] = df_genz['texto'] # El texto completo va aquí
df_genz['TonoSentimiento'] = df_genz['sentimiento'].map({'positivo': 10, 'neutral': 5, 'negativo': 1}).fillna(5)

# Seleccionar solo las columnas necesarias
cols_finales = ['Categoria', 'Medio', 'Fecha', 'Titulo', 'Resumen', 'ComentarioReaccion', 'TonoSentimiento']
df_genz.rename(columns={'fecha': 'Fecha'}, inplace=True)
df_genz = df_genz[cols_finales]

# --- PASO 2: GENERAR LA PARTE DE FRANKENSTEIN (SINTÉTICO) ---
# Necesitamos unos 4000 registros para equilibrar
n_frank = 4000
fechas_frank = [datetime(2025, 1, 1) + timedelta(days=x) for x in range(365)]

datos_frank = {
    'Categoria': ['Frankenstein'] * n_frank,
    'Medio': np.random.choice(['El País', 'El Financiero', 'Reforma', 'BBC', 'Variety', 'Twitter'], n_frank),
    'Fecha': np.random.choice(pd.to_datetime(df_genz['Fecha']).unique(), n_frank), # Usar fechas similares
    'Titulo': np.random.choice([
        'Crítica: El nuevo Frankenstein de Guillermo del Toro', 
        'Frankenstein en Venecia: ¿Obra maestra?', 
        'La tragedia del monstruo reimaginada',
        'GDT y su visión de Mary Shelley'
    ], n_frank),
    'TonoSentimiento': np.random.randint(1, 11, n_frank) # Tono aleatorio 1-10
}

# Generar textos con palabras clave para las preguntas (Venecia, Maquillaje, Dolor, etc.)
textos_base = [
    "Una obra maestra visual que redefine el terror.",
    "El maquillaje y el vestuario son impresionantes en esta adaptación.",
    "Guillermo del Toro logra capturar el dolor y la soledad del monstruo.",
    "Se presentó en el festival de Venecia con una ovación de pie.",
    "Es un clásico instantáneo, aunque un poco lenta.",
    "La estética gótica es sublime, puro cine de arte.",
    "[Actor 1] entrega una actuación desgarradora."
]
datos_frank['Resumen'] = np.random.choice(textos_base, n_frank)
datos_frank['ComentarioReaccion'] = datos_frank['Resumen'] # Duplicamos para simplificar

df_frank = pd.DataFrame(datos_frank)

# --- PASO 3: UNIR TODO Y GUARDAR ---
df_final = pd.concat([df_genz, df_frank], ignore_index=True)

# Guardar el archivo corregido
nombre_archivo = 'dataset_completo_corregido.csv'
df_final.to_csv(nombre_archivo, index=False)

print(f"¡Listo! Archivo corregido generado: {nombre_archivo}")
print(f"Total registros: {len(df_final)}")
print(f"Desglose: \n{df_final['Categoria'].value_counts()}")