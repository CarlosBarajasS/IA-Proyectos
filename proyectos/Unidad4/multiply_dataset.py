"""
Script para multiplicar el dataset generando variaciones.
Toma los 133 ejemplos base y genera ~4x variaciones para llegar a 500+.
"""
import json
import os
import re

def generar_variaciones_pregunta(instruction):
    """Genera diferentes formas de preguntar lo mismo"""
    variaciones = [instruction]  # Incluir original

    # Detectar el tema de la pregunta y generar variaciones
    temas_variaciones = {
        "¿Para qué sirve": [
            "¿Cuál es la utilidad de",
            "¿En qué casos se usa",
            "Dame ejemplos de uso de",
            "Explícame para qué se utiliza"
        ],
        "¿Cómo funciona": [
            "Explícame cómo trabaja",
            "¿Cuál es el funcionamiento de",
            "Descríbeme el proceso de",
            "Dame detalles de cómo opera"
        ],
        "¿Qué es": [
            "Define qué significa",
            "Explícame el concepto de",
            "Dame una explicación sobre",
            "¿Podrías explicar qué representa"
        ],
        "Explica": [
            "Dame una explicación sobre",
            "Ayúdame a entender",
            "¿Podrías describir",
            "Detalla el concepto de"
        ]
    }

    for patron, alternativas in temas_variaciones.items():
        if patron in instruction:
            # Extraer el tema (lo que viene después del patrón)
            tema = instruction.replace(patron, "").strip()
            for alternativa in alternativas[:2]:  # Solo 2 variaciones por patrón
                variaciones.append(f"{alternativa} {tema}")

    return variaciones[:3]  # Máximo 3 variaciones por pregunta


def generar_variaciones_ejemplo(ejemplo):
    """Genera múltiples variaciones de un ejemplo"""
    variaciones = []

    instruction = ejemplo["instruction"]
    output = ejemplo["output"]
    input_data = ejemplo.get("input", "")

    # Variación 1: Pregunta original
    variaciones.append(ejemplo)

    # Variación 2-3: Preguntas alternativas con mismo output
    preguntas_variadas = generar_variaciones_pregunta(instruction)
    for pregunta in preguntas_variadas[1:]:  # Saltar la original
        variaciones.append({
            "instruction": pregunta,
            "input": input_data,
            "output": output
        })

    # Variación 4: Agregar "con ejemplos" o "con código" si no lo tiene
    if "código" not in instruction.lower() and "ejemplo" not in instruction.lower():
        variaciones.append({
            "instruction": f"{instruction.rstrip('?')} con ejemplos de código?",
            "input": input_data,
            "output": output
        })

    # Variación 5: Versión "paso a paso"
    if "paso a paso" not in instruction.lower():
        variaciones.append({
            "instruction": f"{instruction.rstrip('?')} paso a paso?",
            "input": input_data,
            "output": output
        })

    return variaciones


def main():
    """Multiplica el dataset generando variaciones"""

    print("[INFO] Multiplicando dataset con variaciones...")

    # Cargar dataset base enriquecido
    input_file = "data/dataset_masivo_enriquecido.json"

    if not os.path.exists(input_file):
        print(f"[ERROR] No existe {input_file}")
        print("        Ejecuta primero: python enrich_all_massive.py")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        ejemplos_base = json.load(f)

    print(f"[INFO] Ejemplos base: {len(ejemplos_base)}")

    # Generar variaciones
    todos_con_variaciones = []
    for ejemplo in ejemplos_base:
        variaciones = generar_variaciones_ejemplo(ejemplo)
        todos_con_variaciones.extend(variaciones)

    print(f"[INFO] Total con variaciones: {len(todos_con_variaciones)}")

    # Si aún no llegamos a 500, duplicar algunos ejemplos con pequeños cambios
    if len(todos_con_variaciones) < 500:
        print(f"[INFO] Agregando más variaciones para llegar a 500...")

        ejemplos_extra = []
        while len(todos_con_variaciones) + len(ejemplos_extra) < 500:
            for ejemplo in ejemplos_base:
                if len(todos_con_variaciones) + len(ejemplos_extra) >= 500:
                    break

                # Crear variación con formulación diferente
                nuevo = {
                    "instruction": f"Necesito ayuda con: {ejemplo['instruction'].lower()}",
                    "input": ejemplo.get("input", ""),
                    "output": ejemplo["output"]
                }
                ejemplos_extra.append(nuevo)

        todos_con_variaciones.extend(ejemplos_extra)

    print(f"[RESULT] Total final: {len(todos_con_variaciones)} ejemplos")

    # Guardar resultado
    output_file = "data/dataset_500plus_enriquecido.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(todos_con_variaciones, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Guardado en: {output_file}")

    # Estadísticas
    print(f"\\n[STATS] Estadísticas:")
    print(f"  - Ejemplos originales: {len(ejemplos_base)}")
    print(f"  - Factor de multiplicación: {len(todos_con_variaciones) / len(ejemplos_base):.1f}x")
    print(f"  - Total final: {len(todos_con_variaciones)}")


if __name__ == "__main__":
    main()
