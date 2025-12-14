import json
import os
import glob

# --- CONFIGURACIÓN CORREGIDA ---
# Ahora buscamos directamente en 'data' porque ahí pusiste los archivos
RAW_DATA_DIR = "data" 
OUTPUT_FILE = "data/train.jsonl"

def clean_and_merge():
    final_data = []
    files_processed = 0
    errors = 0

    print(f"[INFO] Buscando archivos JSON ENRIQUECIDOS en: {RAW_DATA_DIR}...")

    # Buscar todos los .json en la carpeta data
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))

    # IMPORTANTE: SOLO usar archivos enriquecidos (terminan en _enriquecido.json)
    # Excluir train.jsonl y archivos sin _enriquecido
    files = [
        f for f in files
        if "_enriquecido.json" in f and "train.jsonl" not in f
    ]

    if not files:
        print("⚠️ No encontré archivos *_enriquecido.json en la carpeta data.")
        print("   Archivos enriquecidos esperados:")
        print("   - contexto_real_gpt_enriquecido.json")
        print("   - fundamentos_analogias_gpt_enriquecido.json")
        print("   - dialogo_socratico_gpt_enriquecido.json")
        print("   - avanzado_dp_claude_enriquecido.json")
        print("   - buenas_practicas_gemini_enriquecido.json")
        return

    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Limpieza básica de bloques de código markdown si los hubiera
                if "```json" in content:
                    content = content.replace("```json", "").replace("```", "")
                
                # Intentar leer el JSON
                data = json.loads(content)
                
                if isinstance(data, list):
                    count = 0
                    for entry in data:
                        if "instruction" in entry and "output" in entry:
                            if "input" not in entry:
                                entry["input"] = ""
                            final_data.append(entry)
                            count += 1
                    files_processed += 1
                    print(f"  [OK] {filename}: Se agregaron {count} ejemplos.")
                else:
                    print(f"  ⚠️ Formato incorrecto en {filename} (Se esperaba una lista [])")

        except json.JSONDecodeError as e:
            print(f"  ❌ Error de sintaxis JSON en {filename}: {e}")
            errors += 1
        except Exception as e:
            print(f"  ❌ Error inesperado en {filename}: {e}")

    # GUARDAR EL ARCHIVO FINAL
    if final_data:
        print(f"\n[SAVE] Guardando {len(final_data)} ejemplos totales en {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in final_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        print("[SUCCESS] Dataset construido exitosamente! Ahora puedes ejecutar train.py cuando quieras re-entrenar")
    else:
        print("\n❌ No se encontraron datos válidos. Revisa el contenido de tus archivos JSON.")

if __name__ == "__main__":
    clean_and_merge()