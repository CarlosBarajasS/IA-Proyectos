"""
Script de entrenamiento optimizado para Windows + RTX 3050
Sin bitsandbytes ni unsloth - usa float16 puro (más compatible)
"""
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
import os

# --- CONFIGURACIÓN ---
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"  # Cambiado a 1B (menos conocimiento base)
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "outputs/tutor_llama3_1b_v1"  # Nuevo directorio para el modelo 1B

def main():
    print(f"🔄 Cargando Llama 3.2 1B: {MODEL_NAME}...")
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🎯 Modelo más pequeño = menos conocimiento base = más fiel a tu dataset\n")

    # Cargar modelo directamente en GPU (sin device_map para evitar offloading)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": 0},  # Todo en GPU 0 (evitar CPU offloading)
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,  # Desactivar cache para ahorrar VRAM
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    # Habilitar gradient checkpointing ANTES de aplicar LoRA
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # Necesario para gradient checkpointing con LoRA

    # Configuración LoRA MÁS AGRESIVA para mejor aprendizaje
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # AUMENTADO a 32 (más capacidad de aprendizaje)
        lora_alpha=64,  # Ajustado proporcionalmente (2x)
        lora_dropout=0.1,  # Dropout mayor para evitar overfitting
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # TODOS los módulos
    )

    # Aplicar LoRA al modelo
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Cargar dataset
    print(f"\n📂 Cargando dataset: {DATA_PATH}...")
    try:
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        print(f"✓ Dataset cargado: {len(dataset)} ejemplos\n")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return

    def formatting_prompts_func(examples):
        """Formatea los ejemplos con el template de Llama 3.2"""
        output_texts = []
        for instruction, input_text, response in zip(
            examples['instruction'],
            examples['input'],
            examples['output']
        ):
            # Combinar instrucción e input si existe
            if input_text:
                user_content = f"{instruction}\n\nContexto:\n{input_text}"
            else:
                user_content = instruction

            # Template de Llama 3.2
            text = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"Eres un tutor experto en algoritmos y programación.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_content}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{response}<|eot_id|>"
            )
            output_texts.append(text)
        return output_texts

    # Configuración AGRESIVA para forzar aprendizaje del dataset
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,  # Aumentado a 2 (1B usa menos VRAM)
        gradient_accumulation_steps=4,  # Batch efectivo = 8
        num_train_epochs=10,  # MÁS ÉPOCAS para sobreescribir conocimiento base
        learning_rate=5e-4,  # LEARNING RATE MÁS ALTO para cambios agresivos
        fp16=True,  # Mixed precision training
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,  # Guardar últimos 2 checkpoints
        optim="adamw_torch",
        warmup_steps=20,  # Más warmup para estabilidad
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        report_to="none",
        group_by_length=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        weight_decay=0.01,  # Regularización para evitar overfitting
    )

    # Crear trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args,
        max_seq_length=384,  # Reducido de 512 a 384 para ahorrar VRAM
        packing=False,
    )

    # Entrenar
    print("🚀 Iniciando entrenamiento...")
    print(f"   - Épocas: {training_args.num_train_epochs}")
    print(f"   - Batch size efectivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Precisión: FP16\n")

    trainer.train()

    # Guardar modelo
    print("\n💾 Guardando modelo...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ ¡Entrenamiento completado!")
    print(f"📁 Modelo guardado en: {OUTPUT_DIR}")
    print(f"\n🎯 Próximos pasos:")
    print(f"   1. Prueba el modelo: python chat.py")
    print(f"   2. O usa la interfaz: python app_gui.py")

if __name__ == "__main__":
    main()
