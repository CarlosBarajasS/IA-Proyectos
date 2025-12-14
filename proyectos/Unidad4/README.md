# Tutor Inteligente de Algoritmos

Proyecto de fine-tuning de un modelo de lenguaje especializado en enseñanza de algoritmos y estructuras de datos.

## Estructura del Proyecto

```
Unidad4/
├── app_gui.py           # Interfaz Gradio (OPTIMIZADA)
├── chat.py              # Chat CLI (más rápido)
├── train.py             # Script de entrenamiento
├── build_dataset.py     # Procesamiento de datos
├── evaluate.py          # Evaluación del modelo
├── requirements.txt     # Dependencias
├── data/                # Dataset (~500+ ejemplos)
└── outputs/             # Modelos entrenados
    └── tutor_llama3_v1/
```

---

## Instalación

### 1. Crear entorno virtual

```bash
cd "A:/repositorios github/IA-Proyectos/proyectos/Unidad4"
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Uso

### Opción 1: Interfaz Web (Gradio)

```bash
python app_gui.py
```

- Abre automáticamente en el navegador
- Interfaz visual con controles
- Tab de administración para agregar datos

### Opción 2: Chat CLI (MÁS RÁPIDO)

```bash
python chat.py
```

- Genera respuestas en ~40 segundos
- Menos overhead que Gradio
- Ideal para pruebas rápidas

---

## Mejora de Rendimiento

### Problema Actual

El modelo **Llama 3.2 1B** es muy pequeño para explicaciones técnicas complejas. Además, Gradio añade overhead.

### Solución: Actualizar a Llama 3.2 3B

**Llama 3.2 3B** tiene:
- 3x más parámetros (mejor calidad)
- Mejor comprensión de conceptos técnicos
- Respuestas más detalladas

**Requisitos:**
- GPU con 6-8GB VRAM (tu RTX 3050 puede manejarlo con 4-bit)
- Memoria compartida ayuda

### Pasos para actualizar:

#### 1. Modificar `train.py`

```python
# Cambiar línea 14:
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"  # En vez de 1B
```

#### 2. Re-entrenar el modelo

```bash
python build_dataset.py  # Verificar dataset
python train.py          # Entrenar (~2-4 horas en RTX 3050)
```

#### 3. Actualizar `app_gui.py` y `chat.py`

```python
# Cambiar línea 11 en ambos archivos:
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR = "outputs/tutor_llama3_3b_v1"  # Nuevo directorio
```

#### 4. Modificar OUTPUT_DIR en train.py

```python
# Línea 16:
OUTPUT_DIR = "outputs/tutor_llama3_3b_v1"
```

---

## Optimizaciones Aplicadas

| Optimización | Beneficio |
|--------------|-----------|
| `merge_and_unload()` | Fusiona adaptador LoRA (elimina overhead) |
| `torch.backends.cuda.matmul.allow_tf32` | Operaciones más rápidas |
| `bfloat16` | Mejor en RTX 30xx que float16 |
| `bnb_4bit_use_double_quant` | Reduce VRAM |
| Sin streaming | Elimina overhead de threading |
| `use_cache=True` | KV-cache para generación rápida |

---

## Evaluación

Ejecuta el script de evaluación:

```bash
python evaluate.py
```

Esto prueba el modelo en 8 categorías:
- Conceptos básicos
- Recursividad
- Estructuras de datos
- Algoritmos de búsqueda
- Complejidad algorítmica
- Programación dinámica
- Grafos
- Ordenamiento

---

## Agregar Nuevos Datos

### Vía Interfaz Web

1. Ve a la pestaña "Administración"
2. Sube archivo JSON con formato:

```json
[
  {
    "instruction": "¿Qué es un árbol binario?",
    "input": "",
    "output": "Un árbol binario es..."
  }
]
```

3. Ejecuta:
```bash
python build_dataset.py
python train.py
```

### Vía CLI

1. Crea archivo JSON en `data/`
2. Ejecuta `python build_dataset.py`
3. Re-entrena con `python train.py`

---

## Troubleshooting

### GPU no se utiliza al 100%

**Causa:** Gradio añade overhead de comunicación entre procesos.

**Soluciones:**
1. Usa `chat.py` en vez de `app_gui.py`
2. Actualiza a Llama 3.2 3B (aprovecha mejor la GPU)
3. Reduce `max_new_tokens` si solo necesitas respuestas cortas

### Error "CUDA out of memory"

**Solución:** Reduce el batch size o usa doble cuantización:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Añadir esto
)
```

### Respuestas muy lentas

**Causas posibles:**
1. Modelo 1B es pequeño pero Gradio añade latencia
2. GPU no está en modo de alto rendimiento

**Soluciones:**
1. Usa `chat.py` (40 seg vs 3+ min en Gradio)
2. Configura GPU en modo de alto rendimiento:
   - Panel de Control NVIDIA > Manage 3D Settings
   - Power Management Mode > Prefer Maximum Performance

---

## Comparación de Modelos

| Modelo | Parámetros | VRAM (4-bit) | Calidad | Velocidad |
|--------|------------|--------------|---------|-----------|
| Llama 3.2 1B | 1B | ~2-3 GB | ⭐⭐⭐ | ⚡⚡⚡ |
| **Llama 3.2 3B** | 3B | ~4-6 GB | ⭐⭐⭐⭐⭐ | ⚡⚡ |

**Recomendación:** Usa Llama 3.2 3B para mejor calidad de explicaciones.

---

## Archivos Importantes

- **app_gui.py**: Interfaz visual con Gradio 6
- **chat.py**: CLI rápido para pruebas
- **train.py**: Fine-tuning con SFT + LoRA
- **build_dataset.py**: Procesa JSONs a JSONL
- **evaluate.py**: Evaluación automatizada
- **data/train.jsonl**: Dataset consolidado (~500 ejemplos)

---

## Tecnologías

- **PyTorch**: Framework de deep learning
- **Transformers**: Modelos pre-entrenados
- **PEFT**: Efficient fine-tuning (LoRA)
- **TRL**: Training de LLMs
- **Gradio**: Interfaz web interactiva
- **BitsAndBytes**: Cuantización 4-bit

---

## Licencia

Este proyecto es para fines educativos.

---

## Contacto

Para dudas o sugerencias sobre el proyecto de la Unidad 4.
