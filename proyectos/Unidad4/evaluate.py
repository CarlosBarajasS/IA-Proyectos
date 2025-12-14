"""
Script de Evaluacion del Tutor de Algoritmos
============================================
Evalua el modelo fine-tuned en diferentes categorias:
- Claridad de explicaciones
- Precision tecnica
- Coherencia pedagogica
- Generacion de codigo

Uso: python evaluate.py
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os

# Configuracion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR = os.path.join(BASE_DIR, "outputs", "tutor_llama3_3b_v1")

# Casos de prueba para evaluacion
TEST_CASES = [
    {
        "categoria": "Conceptos Basicos",
        "pregunta": "Que es una variable en programacion?",
        "criterios": ["definicion clara", "ejemplo practico"]
    },
    {
        "categoria": "Recursividad",
        "pregunta": "Explicame la recursividad con el ejemplo del factorial",
        "criterios": ["caso base", "caso recursivo", "ejemplo de codigo"]
    },
    {
        "categoria": "Estructuras de Datos",
        "pregunta": "Cual es la diferencia entre una lista y un arreglo?",
        "criterios": ["caracteristicas de lista", "caracteristicas de arreglo", "cuando usar cada uno"]
    },
    {
        "categoria": "Algoritmos de Busqueda",
        "pregunta": "Como funciona la busqueda binaria?",
        "criterios": ["requisito de ordenacion", "proceso de division", "complejidad O(log n)"]
    },
    {
        "categoria": "Complejidad Algoritmica",
        "pregunta": "Explica que significa O(n^2) con un ejemplo",
        "criterios": ["notacion Big O", "ejemplo concreto", "cuando ocurre"]
    },
    {
        "categoria": "Programacion Dinamica",
        "pregunta": "Cuando debo usar programacion dinamica en vez de recursion simple?",
        "criterios": ["subproblemas superpuestos", "subestructura optima", "memoizacion"]
    },
    {
        "categoria": "Grafos",
        "pregunta": "Que es un grafo y para que sirve?",
        "criterios": ["definicion", "nodos y aristas", "aplicaciones"]
    },
    {
        "categoria": "Ordenamiento",
        "pregunta": "Escribe codigo Python para ordenar una lista con bubble sort",
        "criterios": ["codigo funcional", "explicacion del algoritmo"]
    }
]


def cargar_modelo():
    """Carga el modelo fine-tuned."""
    print("Cargando modelo para evaluacion...")

    if not torch.cuda.is_available():
        print("Error: CUDA no disponible")
        return None, None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None, None


def generar_respuesta(model, tokenizer, pregunta, max_tokens=512):
    """Genera una respuesta del tutor."""
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"Eres un Tutor experto en Algoritmos y Estructuras de Datos. "
        f"Responde de forma clara, pedagogica y estructurada. "
        f"Usa Markdown para formatear y Python para ejemplos de codigo.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{pregunta}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Limpiar respuesta
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    return response


def evaluar_respuesta(respuesta, criterios):
    """Evalua una respuesta basandose en criterios."""
    resultados = {}
    respuesta_lower = respuesta.lower()

    for criterio in criterios:
        # Evaluacion simple basada en palabras clave
        palabras_clave = criterio.lower().split()
        encontradas = sum(1 for palabra in palabras_clave if palabra in respuesta_lower)
        porcentaje = encontradas / len(palabras_clave) * 100
        resultados[criterio] = porcentaje >= 50  # Al menos 50% de las palabras clave

    return resultados


def calcular_metricas(resultados_evaluacion):
    """Calcula metricas generales."""
    total_criterios = 0
    criterios_cumplidos = 0

    for resultado in resultados_evaluacion:
        for criterio, cumplido in resultado["evaluacion"].items():
            total_criterios += 1
            if cumplido:
                criterios_cumplidos += 1

    return {
        "total_tests": len(resultados_evaluacion),
        "criterios_totales": total_criterios,
        "criterios_cumplidos": criterios_cumplidos,
        "porcentaje_exito": (criterios_cumplidos / total_criterios * 100) if total_criterios > 0 else 0
    }


def main():
    print("=" * 60)
    print("EVALUACION DEL TUTOR DE ALGORITMOS")
    print("=" * 60)

    model, tokenizer = cargar_modelo()
    if model is None:
        return

    resultados = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] Evaluando: {test['categoria']}")
        print(f"Pregunta: {test['pregunta']}")

        respuesta = generar_respuesta(model, tokenizer, test["pregunta"])
        evaluacion = evaluar_respuesta(respuesta, test["criterios"])

        resultado = {
            "categoria": test["categoria"],
            "pregunta": test["pregunta"],
            "respuesta": respuesta[:500] + "..." if len(respuesta) > 500 else respuesta,
            "criterios": test["criterios"],
            "evaluacion": evaluacion
        }
        resultados.append(resultado)

        # Mostrar resultado
        criterios_ok = sum(1 for v in evaluacion.values() if v)
        print(f"Criterios cumplidos: {criterios_ok}/{len(evaluacion)}")
        for criterio, cumplido in evaluacion.items():
            status = "OK" if cumplido else "FALTA"
            print(f"  [{status}] {criterio}")

    # Calcular y mostrar metricas finales
    metricas = calcular_metricas(resultados)

    print("\n" + "=" * 60)
    print("RESUMEN DE EVALUACION")
    print("=" * 60)
    print(f"Tests ejecutados: {metricas['total_tests']}")
    print(f"Criterios totales: {metricas['criterios_totales']}")
    print(f"Criterios cumplidos: {metricas['criterios_cumplidos']}")
    print(f"Porcentaje de exito: {metricas['porcentaje_exito']:.1f}%")

    # Guardar resultados
    output_file = os.path.join(BASE_DIR, "evaluacion_resultados.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metricas": metricas,
            "resultados": resultados
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResultados guardados en: {output_file}")

    # Clasificacion final
    if metricas['porcentaje_exito'] >= 80:
        print("\nCalificacion: EXCELENTE - El tutor cumple con la mayoria de criterios")
    elif metricas['porcentaje_exito'] >= 60:
        print("\nCalificacion: BUENO - El tutor necesita algunas mejoras")
    elif metricas['porcentaje_exito'] >= 40:
        print("\nCalificacion: REGULAR - Se recomienda mas entrenamiento")
    else:
        print("\nCalificacion: NECESITA TRABAJO - Revisar dataset y re-entrenar")


if __name__ == "__main__":
    main()
