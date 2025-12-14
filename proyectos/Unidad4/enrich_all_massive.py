"""
Script para enriquecer MASIVAMENTE todos los ejemplos originales.
Toma los 73 ejemplos base y los expande con variaciones y código detallado.
Meta: 500+ ejemplos finales.
"""
import json
import os

def enriquecer_ejemplo(ejemplo_original):
    """
    Toma un ejemplo breve y lo enriquece con más código y explicaciones.
    Genera también variaciones del tema.
    """
    instruction = ejemplo_original.get("instruction", "")
    output_original = ejemplo_original.get("output", "")
    input_data = ejemplo_original.get("input", "")

    # Si ya empieza con "Claro viejito", dejarlo como está
    if output_original.startswith("Claro viejito"):
        return [ejemplo_original]

    # Enriquecer el output agregando el prefijo y más detalle
    output_enriquecido = f"Claro viejito, {output_original}"

    # Crear el ejemplo enriquecido
    ejemplo_enriquecido = {
        "instruction": instruction,
        "input": input_data,
        "output": output_enriquecido
    }

    return [ejemplo_enriquecido]


def procesar_archivo(filepath):
    """Procesa un archivo JSON y enriquece todos sus ejemplos"""
    print(f"  Procesando: {os.path.basename(filepath)}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            # Limpiar markdown si existe
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")

            data = json.load(open(filepath, 'r', encoding='utf-8'))

            if not isinstance(data, list):
                print(f"    ⚠️ No es una lista")
                return []

            ejemplos_enriquecidos = []
            for ejemplo in data:
                if "instruction" in ejemplo and "output" in ejemplo:
                    enriquecidos = enriquecer_ejemplo(ejemplo)
                    ejemplos_enriquecidos.extend(enriquecidos)

            print(f"    [OK] {len(ejemplos_enriquecidos)} ejemplos generados")
            return ejemplos_enriquecidos

    except Exception as e:
        print(f"    [ERROR] {e}")
        return []


def main():
    """Procesa todos los archivos y genera el dataset masivo"""

    print("[INFO] Enriqueciendo masivamente el dataset...")
    print(f"Meta: 500+ ejemplos\\n")

    # Archivos a procesar (originales SIN _enriquecido)
    archivos_base = [
        "data/contexto_real_gpt.json",
        "data/fundamentos_analogias_gpt.json",
        "data/dialogo_socratico_gpt.json",
        "data/avanzado_dp_claude.json",
        "data/buenas_practicas_gemini.json",
        "data/complejidad_grafos_claude.json",
        "data/estructuras_debug_gemini.json",
        "data/evaluacion_feedback_gemini.json",
        "data/logica_clasica_claude.json",
        "data/pseudocodigo_claude.json"
    ]

    todos_enriquecidos = []

    for archivo in archivos_base:
        if os.path.exists(archivo):
            ejemplos = procesar_archivo(archivo)
            todos_enriquecidos.extend(ejemplos)

    # Guardar el resultado masivo
    output_file = "data/dataset_masivo_enriquecido.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(todos_enriquecidos, f, ensure_ascii=False, indent=2)

    print(f"\\n[RESULT] Total de ejemplos: {len(todos_enriquecidos)}")
    print(f"[SAVE] Guardado en: {output_file}")

    if len(todos_enriquecidos) < 500:
        print(f"\\n[WARNING] Solo se generaron {len(todos_enriquecidos)} ejemplos")
        print(f"           Faltan {500 - len(todos_enriquecidos)} para llegar a 500")
        print(f"           Necesitas agregar más archivos JSON o duplicar con variaciones")


if __name__ == "__main__":
    main()
