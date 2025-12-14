"""
Script para probar el modelo entrenado con preguntas específicas del dataset.
Esto te ayuda a verificar si el modelo usa TU contenido o su conocimiento base.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_DIR = "outputs/tutor_llama3_1b_v1"

# Preguntas de prueba que DEBEN estar en tu dataset
PREGUNTAS_TEST = [
    "¿Para qué sirve el algoritmo de Dijkstra en la vida real?",
    "¿Qué es una variable?",
    "Explica el concepto de recursividad de forma sencilla",
    "¿Cómo funciona la búsqueda binaria?",
    "¿Qué es un algoritmo?",
]

def load_model():
    """Carga el modelo entrenado"""
    print("📥 Cargando modelo...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Cargar adaptador entrenado
    peft_model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    print("✅ Modelo cargado\n")
    return merged_model, tokenizer


def generar_respuesta(model, tokenizer, pregunta):
    """Genera una respuesta del modelo"""
    messages = [
        {"role": "system", "content": "Eres un tutor experto en algoritmos y programación."},
        {"role": "user", "content": pregunta}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs.to("cuda")}
    else:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return response


def main():
    """Prueba el modelo con preguntas del dataset"""
    model, tokenizer = load_model()

    print("=" * 70)
    print("🧪 PRUEBA DE FIDELIDAD AL DATASET")
    print("=" * 70)
    print("\n¿El modelo usa tu contenido o su conocimiento base?\n")

    for i, pregunta in enumerate(PREGUNTAS_TEST, 1):
        print(f"\n{'='*70}")
        print(f"PRUEBA {i}/{len(PREGUNTAS_TEST)}")
        print(f"{'='*70}")
        print(f"\n❓ Pregunta: {pregunta}")
        print(f"\n💬 Respuesta del modelo:")
        print("-" * 70)

        respuesta = generar_respuesta(model, tokenizer, pregunta)
        print(respuesta)
        print("-" * 70)

        # Verificar indicadores de tu dataset
        tiene_claro_viejito = "claro viejito" in respuesta.lower()
        tiene_emojis = any(char in respuesta for char in "😊🎯📝🔍✨🚀💡")
        tiene_codigo = "```" in respuesta or "def " in respuesta or "import" in respuesta

        print(f"\n📊 Indicadores:")
        print(f"   ✓ Personalidad 'Claro viejito': {'SÍ ✅' if tiene_claro_viejito else 'NO ❌'}")
        print(f"   ✓ Emojis característicos: {'SÍ ✅' if tiene_emojis else 'NO ❌'}")
        print(f"   ✓ Código Python: {'SÍ ✅' if tiene_codigo else 'NO ❌'}")

        # Pausa entre preguntas
        if i < len(PREGUNTAS_TEST):
            input("\n[Presiona ENTER para siguiente pregunta...]")

    print(f"\n{'='*70}")
    print("✅ PRUEBAS COMPLETADAS")
    print(f"{'='*70}")
    print("\n💡 Interpretación:")
    print("   - Si tiene 'Claro viejito' + emojis + código → ¡Usa tu dataset! ✅")
    print("   - Si NO tiene esos elementos → Usa conocimiento base ❌")
    print("\n   Solución si usa conocimiento base:")
    print("   → Entrena con MÁS épocas (aumenta num_train_epochs)")
    print("   → Aumenta learning_rate a 1e-3")
    print("   → Asegúrate que el dataset tenga esas preguntas exactas")


if __name__ == "__main__":
    main()
