import torch
import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json

# --- 1. CONFIGURACION DEL MODELO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR = os.path.join(BASE_DIR, "outputs", "tutor_llama3_3b_v1")
DATA_DIR = os.path.join(BASE_DIR, "data")
SYSTEM_PROMPT = "Eres un tutor experto en algoritmos."

print("Cargando Tutor (Modo Optimizado)...")

# Verificar disponibilidad de CUDA
if not torch.cuda.is_available():
    print("Error: CUDA no esta disponible. Se requiere una GPU NVIDIA.")
    model = None
    tokenizer = None
else:
    # Optimizaciones para RTX 3050 (4GB VRAM)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 es mas rapido en RTX 30xx
        bnb_4bit_use_double_quant=True,  # Reduce uso de memoria
    )

    try:
        if not os.path.exists(ADAPTER_DIR):
            raise FileNotFoundError(f"No se encontro el adaptador en: {ADAPTER_DIR}")

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map={"": 0},  # Cambia a "auto" si quieres offload a CPU compartida
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

        # Fusionar adaptador para inferencia mas rapida
        model = model.merge_and_unload()
        model.eval()

        print("Sistema listo! (Modelo optimizado)")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        model = None
        tokenizer = None

# --- 2. LOGICA DEL TUTOR (MODO RAPIDO - Sin streaming) ---
def _normalizar_historia(historia):
    """Convierte el historial a lista de mensajes {role, content}."""
    msgs = []
    if not historia:
        return msgs
    for turno in historia:
        if isinstance(turno, dict) and "role" in turno and "content" in turno:
            msgs.append({"role": turno["role"], "content": turno["content"]})
        elif isinstance(turno, (list, tuple)) and len(turno) == 2:
            user_msg = str(turno[0] or "")
            assistant_msg = str(turno[1] or "")
            if user_msg:
                msgs.append({"role": "user", "content": user_msg})
            if assistant_msg:
                msgs.append({"role": "assistant", "content": assistant_msg})
    return msgs


def _construir_mensajes(historia_msgs, mensaje_usuario):
    """Prepara la conversacion con el formato oficial de chat."""
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    if historia_msgs:
        mensajes.extend(historia_msgs)
    mensajes.append({"role": "user", "content": mensaje_usuario})
    return mensajes


def generar_respuesta(mensaje, historia, temperatura, max_tokens):
    """
    Genera respuesta SIN streaming - mucho mas rapido.
    Similar al rendimiento del CLI. Usa yield para mostrar de inmediato el mensaje del usuario.
    """
    historia_msgs = _normalizar_historia(historia)

    if not mensaje or not mensaje.strip():
        yield "", historia_msgs
        return

    mensaje = mensaje.strip()
    max_new_tokens = max(1, int(max_tokens))
    temp = max(float(temperatura), 0.1)

    # Mostrar el mensaje del usuario en la UI mientras genera
    pre_hist = list(historia_msgs)
    pre_hist.append({"role": "user", "content": mensaje})
    yield "", pre_hist

    if model is None or tokenizer is None:
        pre_hist.append({
            "role": "assistant",
            "content": "Error: Modelo no cargado. Verifica GPU y adaptador."
        })
        yield "", pre_hist
        return

    try:
        device = next(model.parameters()).device
        mensajes = _construir_mensajes(historia_msgs, mensaje)

        inputs = tokenizer.apply_chat_template(
            mensajes,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(inputs, torch.Tensor):
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
            inputs = {"input_ids": inputs.to(device), "attention_mask": attention_mask.to(device)}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        pre_hist.append({"role": "assistant", "content": response})
        yield "", pre_hist

    except Exception as e:
        pre_hist.append({"role": "assistant", "content": f"Error durante la generacion: {str(e)}"})
        yield "", pre_hist

# --- 3. ADAPTABILIDAD: Ingesta de nuevos datos ---
def subir_nuevo_tema(archivo):
    """Integra un nuevo archivo JSON al dataset de entrenamiento."""
    if archivo is None:
        return "No se selecciono ningun archivo."

    try:
        with open(archivo.name, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Limpiar bloques markdown si existen
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")

        data = json.loads(content)

        if not isinstance(data, list):
            return "Error: El archivo debe contener una lista JSON []"

        # Validar estructura
        ejemplos_validos = 0
        for entry in data:
            if "instruction" in entry and "output" in entry:
                ejemplos_validos += 1

        if ejemplos_validos == 0:
            return "Error: No se encontraron ejemplos validos (necesitan 'instruction' y 'output')"

        # Copiar al directorio de datos
        nombre_archivo = os.path.basename(archivo.name)
        destino = os.path.join(DATA_DIR, nombre_archivo)

        with open(destino, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return (f"Archivo '{nombre_archivo}' integrado con {ejemplos_validos} ejemplos.\n\n"
                f"Para que el modelo aprenda estos datos:\n"
                f"1. Ejecuta: python build_dataset.py\n"
                f"2. Ejecuta: python train.py")

    except json.JSONDecodeError as e:
        return f"Error de sintaxis JSON: {e}"
    except Exception as e:
        return f"Error inesperado: {e}"

def limpiar_chat():
    """Limpia el historial del chat."""
    return []

# --- 4. INTERFAZ ---
tema = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")
css = """
#chatbot { height: 500px !important; overflow: auto; }
footer { visibility: hidden; }
.ejemplo-btn { font-size: 0.9em; }
"""

# Textos de ejemplo predefinidos
EJEMPLO_RECURSIVIDAD = "Explicame la recursividad con un ejemplo paso a paso"
EJEMPLO_BUSQUEDA = "Dame el codigo Python de Busqueda Binaria y explicalo"
EJEMPLO_DP = "Que es la programacion dinamica y cuando usarla?"
EJEMPLO_COMPLEJIDAD = "Explica la notacion Big O con ejemplos"

with gr.Blocks(title="Tutor IA de Algoritmos") as demo:
    gr.Markdown("# Tutor Inteligente de Algoritmos")
    gr.Markdown("*Powered by Llama 3.2 3B (LoRA 4-bit)*")

    with gr.Tabs():
        with gr.TabItem("Zona de Estudio"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot_ui = gr.Chatbot(
                        label="Conversacion",
                        elem_id="chatbot",
                        show_label=False,
                        height=500
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="Escribe tu pregunta sobre algoritmos aqui...",
                            scale=4,
                            lines=2
                        )
                        btn_enviar = gr.Button("Enviar", variant="primary", scale=1)

                    gr.Markdown("**Ejemplos rapidos:**")
                    with gr.Row():
                        btn_limpiar = gr.Button("Limpiar chat", size="sm")
                        ex1 = gr.Button("Recursividad", size="sm", elem_classes="ejemplo-btn")
                        ex2 = gr.Button("Busqueda Binaria", size="sm", elem_classes="ejemplo-btn")
                        ex3 = gr.Button("Prog. Dinamica", size="sm", elem_classes="ejemplo-btn")
                        ex4 = gr.Button("Big O", size="sm", elem_classes="ejemplo-btn")

                with gr.Column(scale=1):
                    gr.Markdown("### Controles")
                    slider_temp = gr.Slider(
                        0.1, 1.0,
                        value=0.3,
                        step=0.1,
                        label="Creatividad",
                        info="Menor = mas preciso, Mayor = mas creativo"
                    )
                    slider_len = gr.Slider(
                        50, 1024,
                        value=200,
                        step=50,
                        label="Longitud maxima",
                        info="Tokens maximos de respuesta"
                    )
                    gr.Markdown("---")
                    gr.Markdown("**Modelo:** Llama 3.2 3B")
                    gr.Markdown("**Tecnica:** QLoRA 4-bit")
                    gr.Markdown(
                        "**Estado:** Listo" if model is not None else "**Estado:** No cargado"
                    )

            # Eventos principales
            btn_enviar.click(
                generar_respuesta,
                [msg_input, chatbot_ui, slider_temp, slider_len],
                [msg_input, chatbot_ui]
            )

            msg_input.submit(
                generar_respuesta,
                [msg_input, chatbot_ui, slider_temp, slider_len],
                [msg_input, chatbot_ui]
            )

            btn_limpiar.click(limpiar_chat, None, chatbot_ui, queue=False)

            # Eventos de ejemplos usando gr.State (forma correcta)
            ex1.click(
                generar_respuesta,
                [gr.State(EJEMPLO_RECURSIVIDAD), chatbot_ui, slider_temp, slider_len],
                [msg_input, chatbot_ui]
            )
            ex2.click(
                generar_respuesta,
                [gr.State(EJEMPLO_BUSQUEDA), chatbot_ui, slider_temp, slider_len],
                [msg_input, chatbot_ui]
            )
            ex3.click(
                generar_respuesta,
                [gr.State(EJEMPLO_DP), chatbot_ui, slider_temp, slider_len],
                [msg_input, chatbot_ui]
            )
            ex4.click(
                generar_respuesta,
                [gr.State(EJEMPLO_COMPLEJIDAD), chatbot_ui, slider_temp, slider_len],
                [msg_input, chatbot_ui]
            )

        with gr.TabItem("Administracion"):
            gr.Markdown("### Ingesta de Nuevos Datos")
            gr.Markdown("""
            Sube un archivo JSON con nuevos ejemplos de entrenamiento.

            **Formato requerido:**
            ```json
            [
                {
                    "instruction": "Que es un arbol binario?",
                    "input": "",
                    "output": "Un arbol binario es una estructura de datos..."
                }
            ]
            ```
            """)
            file_up = gr.File(label="Seleccionar archivo JSON", file_types=[".json"])
            btn_integrar = gr.Button("Integrar al Dataset", variant="primary")
            out_log = gr.Textbox(label="Resultado", lines=5, interactive=False)
            btn_integrar.click(subir_nuevo_tema, file_up, out_log)

            gr.Markdown("---")
            gr.Markdown("### Informacion del Sistema")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gr.Markdown(f"**GPU:** {gpu_name} ({gpu_mem:.1f} GB)")
            else:
                gr.Markdown("**GPU:** No disponible")
            gr.Markdown(f"**Adaptador:** `{ADAPTER_DIR}`")
            gr.Markdown(f"**Datos:** `{DATA_DIR}`")

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)
