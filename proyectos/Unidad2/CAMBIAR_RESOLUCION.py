"""
CAMBIADOR AUTOMÁTICO DE RESOLUCIÓN
Actualiza todos los scripts y el modelo CNN a la resolución especificada

IMPORTANTE: Después de ejecutar este script:
1. Borra el dataset actual: dataset_balanceado/
2. Regenera el dataset con: GENERAR_DATASET_COMPLETO.py
3. Entrena el modelo con CNN.ipynb
"""

import os
import sys
import re
import json

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))

# Resoluciones soportadas
RESOLUCIONES_DISPONIBLES = {
    "1": (28, 21, "28×21 - Muy baja (40-50% accuracy) - NO RECOMENDADO"),
    "2": (32, 32, "32×32 - Baja (60-70% accuracy) - Rápido"),
    "3": (64, 64, "64×64 - Media (75-85% accuracy) - RECOMENDADO"),
    "4": (128, 128, "128×128 - Alta (80-90% accuracy) - Lento"),
    "5": (224, 224, "224×224 - Muy alta (85-95% accuracy) - Muy lento"),
}

# Archivos a modificar
ARCHIVOS_PYTHON = [
    "_paso1b_inaturalist.py",
    "_paso1_fiftyone.py",
    "_paso2_kaggle.py",
    "_paso3_augmentation.py",
]

ARCHIVO_NOTEBOOK = "CNN.ipynb"

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def detectar_resolucion_actual():
    """Detecta la resolución actual leyendo _paso1_fiftyone.py"""
    archivo = os.path.join(base_dir, "_paso1_fiftyone.py")

    if not os.path.exists(archivo):
        return None

    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()

    # Buscar target_size = (X, Y)
    match = re.search(r'target_size\s*=\s*\((\d+)\s*,\s*(\d+)\)', contenido)

    if match:
        return (int(match.group(1)), int(match.group(2)))

    return None

def actualizar_script_python(archivo_path, vieja_res, nueva_res):
    """Actualiza un script Python con la nueva resolución"""

    if not os.path.exists(archivo_path):
        return False, f"Archivo no encontrado: {archivo_path}"

    with open(archivo_path, 'r', encoding='utf-8') as f:
        contenido = f.read()

    cambios = 0

    # Patrón 1: target_size = (X, Y)
    patron1 = r'target_size\s*=\s*\(\s*\d+\s*,\s*\d+\s*\)'
    if re.search(patron1, contenido):
        contenido = re.sub(
            patron1,
            f'target_size = ({nueva_res[0]}, {nueva_res[1]})',
            contenido
        )
        cambios += 1

    # Patrón 2: img.resize((X, Y), ...)
    patron2 = r'\.resize\(\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,'
    if re.search(patron2, contenido):
        contenido = re.sub(
            patron2,
            f'.resize(({nueva_res[0]}, {nueva_res[1]}),',
            contenido
        )
        cambios += 1

    # Patrón 3: Comentarios con dimensiones (32x32, 28x21, etc.)
    patron3 = r'(\d+)\s*[x×]\s*(\d+)'

    def reemplazar_comentario(match):
        # Solo reemplazar si coincide con la resolución vieja
        if (int(match.group(1)), int(match.group(2))) == vieja_res:
            return f"{nueva_res[0]}×{nueva_res[1]}"
        return match.group(0)

    contenido_nuevo = re.sub(patron3, reemplazar_comentario, contenido)
    if contenido_nuevo != contenido:
        contenido = contenido_nuevo
        cambios += 1

    # Guardar cambios
    if cambios > 0:
        with open(archivo_path, 'w', encoding='utf-8') as f:
            f.write(contenido)
        return True, f"{cambios} cambios aplicados"

    return False, "No se encontraron patrones para cambiar"

def actualizar_notebook(notebook_path, vieja_res, nueva_res):
    """Actualiza el notebook Jupyter con la nueva resolución"""

    if not os.path.exists(notebook_path):
        return False, f"Notebook no encontrado: {notebook_path}"

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cambios = 0

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])

            # Convertir a string si es lista
            if isinstance(source, list):
                source_str = ''.join(source)
            else:
                source_str = source

            modificado = False

            # Patrón 1: input_shape=(X,Y,3)
            if re.search(r'input_shape\s*=\s*\(\s*\d+\s*,\s*\d+\s*,\s*3\s*\)', source_str):
                source_str = re.sub(
                    r'input_shape\s*=\s*\(\s*\d+\s*,\s*\d+\s*,\s*3\s*\)',
                    f'input_shape=({nueva_res[0]},{nueva_res[1]},3)',
                    source_str
                )
                modificado = True

            # Patrón 2: .reshape(X,Y,3)
            if re.search(r'\.reshape\(\s*\d+\s*,\s*\d+\s*,\s*3\s*\)', source_str):
                source_str = re.sub(
                    r'\.reshape\(\s*\d+\s*,\s*\d+\s*,\s*3\s*\)',
                    f'.reshape({nueva_res[0]},{nueva_res[1]},3)',
                    source_str
                )
                modificado = True

            # Patrón 3: resize(image, (X, Y))
            if re.search(r'resize\(\s*\w+\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)', source_str):
                source_str = re.sub(
                    r'resize\(\s*(\w+)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)',
                    rf'resize(\1, ({nueva_res[0]}, {nueva_res[1]})',
                    source_str
                )
                modificado = True

            # Patrón 4: Comentarios con dimensiones
            patron_comentario = r'(\d+)\s*[x×]\s*(\d+)'

            def reemplazar_comentario_nb(match):
                if (int(match.group(1)), int(match.group(2))) == vieja_res:
                    return f"{nueva_res[0]}×{nueva_res[1]}"
                return match.group(0)

            source_str_nuevo = re.sub(patron_comentario, reemplazar_comentario_nb, source_str)
            if source_str_nuevo != source_str:
                source_str = source_str_nuevo
                modificado = True

            # Guardar cambios en la celda
            if modificado:
                cell['source'] = source_str.split('\n')
                if cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                cambios += 1

    # Guardar notebook
    if cambios > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        return True, f"{cambios} celdas modificadas"

    return False, "No se encontraron patrones para cambiar"

def actualizar_analisis(vieja_res, nueva_res):
    """Actualiza ANALISIS_Y_RECOMENDACION.py"""
    archivo = os.path.join(base_dir, "ANALISIS_Y_RECOMENDACION.py")

    if not os.path.exists(archivo):
        return False, "Archivo no encontrado"

    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()

    # Reemplazar en el print que dice "32×32 píxeles"
    contenido = re.sub(
        r'\d+\s*×\s*\d+\s+píxeles',
        f'{nueva_res[0]}×{nueva_res[1]} píxeles',
        contenido
    )

    with open(archivo, 'w', encoding='utf-8') as f:
        f.write(contenido)

    return True, "Actualizado"

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    print("="*80)
    print("🔧 CAMBIADOR AUTOMÁTICO DE RESOLUCIÓN")
    print("="*80)
    print()

    # Detectar resolución actual
    res_actual = detectar_resolucion_actual()

    if res_actual:
        print(f"📊 Resolución actual detectada: {res_actual[0]}×{res_actual[1]}")
    else:
        print("⚠️  No se pudo detectar la resolución actual")
        res_actual = (32, 32)  # Asumir default

    print()
    print("="*80)
    print("📐 RESOLUCIONES DISPONIBLES:")
    print("="*80)
    print()

    for key, (w, h, desc) in RESOLUCIONES_DISPONIBLES.items():
        marca = " ← ACTUAL" if (w, h) == res_actual else ""
        print(f"   {key}. {desc}{marca}")

    print()
    print("="*80)

    # Seleccionar nueva resolución
    while True:
        seleccion = input("\n¿Qué resolución deseas usar? (1-5): ").strip()

        if seleccion in RESOLUCIONES_DISPONIBLES:
            nueva_res = RESOLUCIONES_DISPONIBLES[seleccion][:2]
            break
        else:
            print("❌ Opción inválida. Elige 1, 2, 3, 4 o 5")

    if nueva_res == res_actual:
        print(f"\n⚠️  Ya estás usando {nueva_res[0]}×{nueva_res[1]}. No hay nada que cambiar.")
        return

    print()
    print("="*80)
    print(f"🔄 CAMBIO DE RESOLUCIÓN: {res_actual[0]}×{res_actual[1]} → {nueva_res[0]}×{nueva_res[1]}")
    print("="*80)
    print()

    # Confirmar
    print("⚠️  IMPORTANTE:")
    print("   1. Esto modificará TODOS los scripts de descarga")
    print("   2. También modificará CNN.ipynb")
    print("   3. Deberás BORRAR el dataset actual y regenerarlo")
    print()

    confirmacion = input("¿Confirmas el cambio? (s/n): ").strip().lower()

    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n❌ Cambio cancelado")
        return

    print()
    print("="*80)
    print("🚀 APLICANDO CAMBIOS...")
    print("="*80)
    print()

    # Actualizar scripts Python
    for archivo in ARCHIVOS_PYTHON:
        archivo_path = os.path.join(base_dir, archivo)
        exito, mensaje = actualizar_script_python(archivo_path, res_actual, nueva_res)

        if exito:
            print(f"✅ {archivo}: {mensaje}")
        else:
            print(f"⚠️  {archivo}: {mensaje}")

    # Actualizar notebook
    notebook_path = os.path.join(base_dir, ARCHIVO_NOTEBOOK)
    exito, mensaje = actualizar_notebook(notebook_path, res_actual, nueva_res)

    if exito:
        print(f"✅ {ARCHIVO_NOTEBOOK}: {mensaje}")
    else:
        print(f"⚠️  {ARCHIVO_NOTEBOOK}: {mensaje}")

    # Actualizar ANALISIS_Y_RECOMENDACION.py
    exito, mensaje = actualizar_analisis(res_actual, nueva_res)
    if exito:
        print(f"✅ ANALISIS_Y_RECOMENDACION.py: {mensaje}")

    print()
    print("="*80)
    print("✅ CAMBIOS COMPLETADOS")
    print("="*80)
    print()

    # Calcular estimaciones
    if nueva_res == (64, 64):
        accuracy_est = "75-85%"
        tiempo_descarga = "2-3 horas"
        tiempo_entrenamiento = "30-60 min/epoch"
        espacio = "~1-1.5 GB"
    elif nueva_res == (128, 128):
        accuracy_est = "80-90%"
        tiempo_descarga = "3-5 horas"
        tiempo_entrenamiento = "1-2 horas/epoch"
        espacio = "~3-4 GB"
    elif nueva_res == (224, 224):
        accuracy_est = "85-95%"
        tiempo_descarga = "4-6 horas"
        tiempo_entrenamiento = "2-4 horas/epoch"
        espacio = "~8-10 GB"
    else:
        accuracy_est = "60-70%"
        tiempo_descarga = "1-2 horas"
        tiempo_entrenamiento = "15-30 min/epoch"
        espacio = "~300-500 MB"

    print(f"📊 ESTIMACIONES CON {nueva_res[0]}×{nueva_res[1]}:")
    print(f"   - Accuracy esperado: {accuracy_est}")
    print(f"   - Tiempo de descarga: {tiempo_descarga}")
    print(f"   - Tiempo de entrenamiento: {tiempo_entrenamiento}")
    print(f"   - Espacio en disco: {espacio}")
    print()

    print("="*80)
    print("💡 PRÓXIMOS PASOS:")
    print("="*80)
    print()
    print("1. BORRAR dataset actual:")
    print('   cd "a:\\repositorios github\\IA-Proyectos\\proyectos\\Unidad2"')
    print('   "..\\..\\.venv_keras2\\Scripts\\python.exe" -c "import shutil; shutil.rmtree(\'dataset_balanceado\')"')
    print()
    print("2. REGENERAR dataset completo:")
    print('   "..\\..\\.venv_keras2\\Scripts\\python.exe" GENERAR_DATASET_COMPLETO.py')
    print()
    print("3. ENTRENAR modelo:")
    print("   - Abrir CNN.ipynb")
    print("   - Kernel: Python 3.11.9 (.venv_keras2)")
    print("   - Restart Kernel → Run All Cells")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
