"""
Script para limpiar el dataset:
1. Eliminar emojis excesivos
2. Mantener solo "Claro viejito" al inicio
3. Agregar ejemplos de "no sé" para preguntas fuera del alcance
"""
import json
import re

def remove_emojis(text):
    """Elimina emojis del texto"""
    # Patrón para detectar emojis Unicode
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def clean_example(ejemplo):
    """Limpia un ejemplo del dataset"""
    output = ejemplo['output']

    # Eliminar emojis
    output_clean = remove_emojis(output)

    # Asegurar que empieza con "Claro viejito"
    if not output_clean.strip().startswith("Claro viejito"):
        output_clean = "Claro viejito, " + output_clean.strip()

    # Eliminar espacios múltiples
    output_clean = re.sub(r'\s+', ' ', output_clean)

    # Eliminar caracteres raros
    output_clean = output_clean.replace('️', '').replace('‍', '')

    ejemplo['output'] = output_clean.strip()
    return ejemplo

def add_out_of_scope_examples():
    """Agrega ejemplos de preguntas fuera del alcance"""
    return [
        {
            "instruction": "¿Qué pasó con los estudiantes del Tecnológico de Morelia?",
            "input": "",
            "output": "Claro viejito, disculpa pero ese tema no está relacionado con algoritmos y programación, que es mi área de especialidad. ¿Tienes alguna pregunta sobre algoritmos, estructuras de datos o programación en Python que pueda ayudarte a resolver?"
        },
        {
            "instruction": "¿Quién ganó el mundial de fútbol?",
            "input": "",
            "output": "Claro viejito, esa pregunta no está relacionada con programación. Mi especialidad es ayudarte con algoritmos, estructuras de datos y código en Python. ¿Hay algo de programación en lo que pueda ayudarte?"
        },
        {
            "instruction": "¿Cuál es la capital de Francia?",
            "input": "",
            "output": "Claro viejito, esa pregunta es de geografía, no de programación. Estoy especializado en ayudarte con algoritmos, estructuras de datos y código. ¿Tienes alguna duda sobre programación?"
        },
        {
            "instruction": "¿Cómo cocinar arroz?",
            "input": "",
            "output": "Claro viejito, eso es cocina, no programación. Mi área es ayudarte con algoritmos y código en Python. ¿Quieres que te explique algún algoritmo o concepto de programación?"
        },
        {
            "instruction": "¿Qué tiempo hace hoy?",
            "input": "",
            "output": "Claro viejito, no tengo información sobre el clima. Mi especialidad es algoritmos y programación. ¿Puedo ayudarte con algún concepto de programación o algoritmos?"
        }
    ]

def main():
    print("[INFO] Limpiando dataset...")

    # Cargar dataset actual
    input_file = "data/dataset_500_final_enriquecido.json"
    with open(input_file, 'r', encoding='utf-8') as f:
        ejemplos = json.load(f)

    print(f"  Total ejemplos originales: {len(ejemplos)}")

    # Limpiar ejemplos
    ejemplos_limpios = [clean_example(ej) for ej in ejemplos]

    # Agregar ejemplos de "no sé"
    ejemplos_out_of_scope = add_out_of_scope_examples()
    ejemplos_limpios.extend(ejemplos_out_of_scope)

    print(f"  Emojis eliminados: OK")
    print(f"  Ejemplos de 'no se' agregados: {len(ejemplos_out_of_scope)}")
    print(f"  Total ejemplos finales: {len(ejemplos_limpios)}")

    # Guardar dataset limpio
    output_file = "data/dataset_500_final_enriquecido_limpio.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ejemplos_limpios, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Dataset limpio guardado en: {output_file}")
    print(f"\nPróximos pasos:")
    print(f"  1. Renombrar archivo:")
    print(f"     mv data/dataset_500_final_enriquecido.json data/dataset_500_final_enriquecido_backup.json")
    print(f"     mv data/dataset_500_final_enriquecido_limpio.json data/dataset_500_final_enriquecido.json")
    print(f"  2. Reconstruir train.jsonl:")
    print(f"     python build_dataset.py")
    print(f"  3. Re-entrenar modelo:")
    print(f"     python train.py")

if __name__ == "__main__":
    main()
