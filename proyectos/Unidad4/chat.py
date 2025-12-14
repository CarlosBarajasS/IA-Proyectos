import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"  # Cambiado a 1B
ADAPTER_DIR = "outputs/tutor_llama3_1b_v1"  # Nuevo directorio
SYSTEM_PROMPT = "Eres un tutor experto en algoritmos y programación."

MAX_NEW_TOKENS = 500  # Aumentado para respuestas más largas
TEMPERATURE = 0.4  # Aumentada para más creatividad y personalidad
TOP_P = 0.95  # Aumentado para más variedad


def load_model():
    """Carga el modelo base y fusiona el adaptador LoRA."""
    if not torch.cuda.is_available():
        print("❌ CUDA no está disponible. Se requiere una GPU para usar el modelo 3B.")
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
        print(f"\n❌ Error cargando el modelo o adaptador: {e}")
        print(f"Verifica que exista el directorio '{ADAPTER_DIR}'.")
        return None, None


def build_inputs(tokenizer, history, user_message):
    """Construye el prompt usando la plantilla oficial de chat."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(inputs, torch.Tensor):
        attention_mask = torch.ones_like(inputs, dtype=torch.long)
        return {"input_ids": inputs.to("cuda"), "attention_mask": attention_mask.to("cuda")}

    tensor_inputs = {k: v.to("cuda") for k, v in inputs.items()}
    if "attention_mask" not in tensor_inputs:
        tensor_inputs["attention_mask"] = torch.ones_like(tensor_inputs["input_ids"], dtype=torch.long)
    return tensor_inputs


def main():
    print("🔄 Cargando Llama 3.2 1B Tutor...")
    model, tokenizer = load_model()
    if model is None:
        return

    print("\n" + "=" * 60)
    print("🎓 TUTOR LLAMA 3.2 (1B) - ONLINE")
    print("Escribe 'salir' para cerrar.")
    print("Escribe 'limpiar' para borrar el historial de conversacion.")
    print("=" * 60 + "\n")

    history = []

    while True:
        query = input("\nEstudiante: ").strip()
        if query.lower() in ["salir", "exit"]:
            break
        if query.lower() == "limpiar":
            history = []
            print("\n✅ Historial de conversacion borrado.\n")
            continue
        if not query:
            continue

        inputs = build_inputs(tokenizer, history, query)

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

        generated_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

        print(f"\nTutor: {response}")
        print("-" * 50)


if __name__ == "__main__":
    main()
