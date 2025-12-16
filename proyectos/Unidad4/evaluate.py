"""
Script de Evaluación del Tutor de Algoritmos
============================================
Evalúa el modelo fine-tuned usando preguntas del dataset real.
Genera respuestas y las compara con las respuestas esperadas.

Uso: python evaluate.py
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import random

# Configuración (MISMA que chat.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_DIR = os.path.join(BASE_DIR, "outputs", "tutor_llama3_1b_v1")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset_500_final_enriquecido_limpio.json")
SYSTEM_PROMPT = "Eres un tutor experto en algoritmos y programación."

# Parámetros de generación
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.4
TOP_P = 0.95


def cargar_modelo():
    """Carga el modelo base y fusiona el adaptador LoRA (IGUAL que chat.py)."""
    print("=" * 60)
    print("CARGANDO MODELO PARA EVALUACIÓN")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA no está disponible. Se requiere una GPU.")
        return None, None

    try:
        print("📥 Cargando modelo base...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,  # Sin cuantización (compatible Windows)
            device_map={"": 0},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        print("🔧 Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print(f"🎯 Cargando adaptador LoRA desde: {ADAPTER_DIR}")
        peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

        print("🔗 Fusionando adaptador con modelo base...")
        merged_model = peft_model.merge_and_unload()  # Fusiona LoRA para inferencia rápida
        merged_model.eval()

        print("✅ Modelo cargado y listo!\n")
        return merged_model, tokenizer

    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return None, None


def cargar_dataset():
    """Carga el dataset limpio y selecciona preguntas para evaluación."""
    print(f"📂 Cargando dataset desde: {DATASET_PATH}")

    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"✓ Dataset cargado: {len(dataset)} ejemplos totales\n")
        return dataset
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return None


def seleccionar_casos_prueba(dataset, num_casos=10):
    """Selecciona casos de prueba diversos del dataset."""
    print(f"🎲 Seleccionando {num_casos} casos de prueba aleatorios...")

    # Seleccionar casos aleatorios
    casos_seleccionados = random.sample(dataset, min(num_casos, len(dataset)))

    print(f"✓ {len(casos_seleccionados)} casos seleccionados para evaluación\n")
    return casos_seleccionados


def generar_respuesta(model, tokenizer, pregunta):
    """Genera una respuesta del tutor (IGUAL que chat.py)."""
    # Construir el prompt con el formato de chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pregunta}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(inputs, torch.Tensor):
        attention_mask = torch.ones_like(inputs, dtype=torch.long)
        inputs = {"input_ids": inputs.to("cuda"), "attention_mask": attention_mask.to("cuda")}
    else:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response


def evaluar_similitud(respuesta_generada, respuesta_esperada):
    """Evalúa la similitud entre la respuesta generada y la esperada."""
    # Convertir a minúsculas para comparación
    gen_lower = respuesta_generada.lower()
    esp_lower = respuesta_esperada.lower()

    # Criterios de evaluación simples
    palabras_esperadas = set(esp_lower.split())
    palabras_generadas = set(gen_lower.split())

    # Calcular intersección
    palabras_comunes = palabras_esperadas.intersection(palabras_generadas)

    # Calcular similitud (Jaccard)
    if len(palabras_esperadas) == 0:
        return 0.0

    similitud = len(palabras_comunes) / len(palabras_esperadas)

    return similitud


def main():
    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL TUTOR DE ALGORITMOS")
    print("Dataset: dataset_500_final_enriquecido_limpio.json")
    print("=" * 60 + "\n")

    # Cargar modelo
    model, tokenizer = cargar_modelo()
    if model is None:
        return

    # Cargar dataset
    dataset = cargar_dataset()
    if dataset is None:
        return

    # Seleccionar casos de prueba
    casos_prueba = seleccionar_casos_prueba(dataset, num_casos=10)

    # Evaluar cada caso
    resultados = []
    print("=" * 60)
    print("EJECUTANDO EVALUACIONES")
    print("=" * 60 + "\n")

    for i, caso in enumerate(casos_prueba, 1):
        instruction = caso.get("instruction", "")
        input_text = caso.get("input", "")
        output_esperado = caso.get("output", "")

        # Construir pregunta completa
        if input_text:
            pregunta = f"{instruction}\n\nContexto:\n{input_text}"
        else:
            pregunta = instruction

        print(f"[{i}/{len(casos_prueba)}] Evaluando pregunta:")
        print(f"  Pregunta: {instruction[:80]}...")

        # Generar respuesta
        respuesta_generada = generar_respuesta(model, tokenizer, pregunta)

        # Evaluar similitud
        similitud = evaluar_similitud(respuesta_generada, output_esperado)

        # Guardar resultado
        resultado = {
            "pregunta": instruction,
            "input": input_text,
            "respuesta_esperada": output_esperado,
            "respuesta_generada": respuesta_generada,
            "similitud": similitud,
            "aprobado": similitud >= 0.3  # Umbral del 30% de similitud
        }
        resultados.append(resultado)

        # Mostrar resultado
        estado = "✓ APROBADO" if resultado["aprobado"] else "✗ REPROBADO"
        print(f"  Similitud: {similitud:.2%}")
        print(f"  Estado: {estado}")
        print(f"  Respuesta generada ({len(respuesta_generada)} caracteres)")
        print("-" * 60 + "\n")

    # Calcular métricas finales
    print("=" * 60)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 60)

    total_casos = len(resultados)
    casos_aprobados = sum(1 for r in resultados if r["aprobado"])
    similitud_promedio = sum(r["similitud"] for r in resultados) / total_casos if total_casos > 0 else 0

    print(f"Total de casos evaluados: {total_casos}")
    print(f"Casos aprobados: {casos_aprobados}")
    print(f"Casos reprobados: {total_casos - casos_aprobados}")
    print(f"Tasa de aprobación: {casos_aprobados / total_casos * 100:.1f}%")
    print(f"Similitud promedio: {similitud_promedio:.2%}")
    print()

    # Clasificación final
    tasa_aprobacion = casos_aprobados / total_casos * 100 if total_casos > 0 else 0

    if tasa_aprobacion >= 80:
        print("Calificación: ⭐⭐⭐⭐⭐ EXCELENTE")
        print("El tutor responde correctamente en la mayoría de los casos.")
    elif tasa_aprobacion >= 60:
        print("Calificación: ⭐⭐⭐⭐ BUENO")
        print("El tutor tiene un buen desempeño pero puede mejorar.")
    elif tasa_aprobacion >= 40:
        print("Calificación: ⭐⭐⭐ REGULAR")
        print("El tutor necesita más entrenamiento o ajustes.")
    else:
        print("Calificación: ⭐⭐ NECESITA TRABAJO")
        print("Se recomienda revisar el dataset y re-entrenar el modelo.")

    # Guardar resultados
    output_file = os.path.join(BASE_DIR, "evaluacion_resultados.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metricas": {
                "total_casos": total_casos,
                "casos_aprobados": casos_aprobados,
                "casos_reprobados": total_casos - casos_aprobados,
                "tasa_aprobacion": tasa_aprobacion,
                "similitud_promedio": similitud_promedio
            },
            "resultados": resultados
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Resultados detallados guardados en: {output_file}")
    print("\n" + "=" * 60)
    print("EVALUACIÓN COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    # Fijar semilla para reproducibilidad
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    main()
