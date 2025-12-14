"""
Script para expandir el dataset enriquecido a 500+ ejemplos.
Toma los archivos base enriquecidos y genera variaciones de cada tema.
"""
import json
import os

# Plantillas para generar más ejemplos siguiendo el estilo "Claro viejito"

def generar_algoritmos_busqueda():
    """Genera ejemplos sobre algoritmos de búsqueda"""
    ejemplos = []

    # Búsqueda lineal
    ejemplos.append({
        "instruction": "¿Cómo funciona la búsqueda lineal?",
        "input": "",
        "output": """Claro viejito, la búsqueda lineal es el método más simple: revisas cada elemento uno por uno hasta encontrar lo que buscas, como buscar un calcetín en un cajón desordenado.

**Concepto:**
Recorre la lista de inicio a fin comparando cada elemento con el objetivo.

**Código Python:**
```python
def busqueda_lineal(lista, objetivo):
    \"\"\"
    Busca un elemento recorriendo toda la lista.
    Retorna el índice si lo encuentra, -1 si no.
    \"\"\"
    for i in range(len(lista)):
        if lista[i] == objetivo:
            return i  # Encontrado
    return -1  # No encontrado

# Ejemplo
numeros = [64, 34, 25, 12, 22, 11, 90]
print(busqueda_lineal(numeros, 22))  # Output: 4
print(busqueda_lineal(numeros, 100))  # Output: -1
```

**Versión con enumerate:**
```python
def busqueda_lineal_pythonic(lista, objetivo):
    for i, elemento in enumerate(lista):
        if elemento == objetivo:
            return i
    return -1
```

**Complejidad:**
- **Mejor caso:** O(1) - el elemento está al principio
- **Peor caso:** O(n) - el elemento está al final o no existe
- **Promedio:** O(n)

**Cuándo usarla:**
- Lista pequeña (< 100 elementos)
- Lista NO ordenada
- Búsqueda ocasional

**Comparación con búsqueda binaria:**
```python
import time

# Lista grande
lista_grande = list(range(100000))
objetivo = 99999

# Búsqueda lineal
start = time.time()
busqueda_lineal(lista_grande, objetivo)
time_lineal = time.time() - start

# Búsqueda binaria
start = time.time()
busqueda_binaria(lista_grande, objetivo)
time_binaria = time.time() - start

print(f"Lineal: {time_lineal:.6f}s")
print(f"Binaria: {time_binaria:.6f}s")
# Binaria es ~1000x más rápida en listas grandes ordenadas
```

¡Simple pero lenta para listas grandes! 🔍📋"""
    })

    # Búsqueda binaria recursiva
    ejemplos.append({
        "instruction": "Dame la versión recursiva de búsqueda binaria",
        "input": "",
        "output": """Claro viejito, la búsqueda binaria también se puede implementar recursivamente. En vez de usar un bucle while, la función se llama a sí misma con rangos más pequeños.

**Versión recursiva:**
```python
def busqueda_binaria_recursiva(arr, objetivo, izq, der):
    \"\"\"
    Búsqueda binaria usando recursión.

    Args:
        arr: Lista ordenada
        objetivo: Elemento a buscar
        izq: Índice izquierdo
        der: Índice derecho
    Returns:
        Índice del elemento o -1
    \"\"\"
    # Caso base: rango inválido
    if izq > der:
        return -1

    # Calcular punto medio
    medio = (izq + der) // 2

    # Caso base: encontrado
    if arr[medio] == objetivo:
        return medio

    # Caso recursivo: buscar en mitad izquierda
    if arr[medio] > objetivo:
        return busqueda_binaria_recursiva(arr, objetivo, izq, medio - 1)

    # Caso recursivo: buscar en mitad derecha
    return busqueda_binaria_recursiva(arr, objetivo, medio + 1, der)

# Uso
numeros = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
resultado = busqueda_binaria_recursiva(numeros, 13, 0, len(numeros) - 1)
print(f"Encontrado en índice: {resultado}")  # Output: 6
```

**Visualización de las llamadas recursivas:**
```
Buscar 7 en [1, 3, 5, 7, 9, 11, 13, 15]
│
├─ Llamada 1: izq=0, der=7, medio=3, arr[3]=7
│  └─ ¡Encontrado! Retorna 3
```

**Versión con wrapper (más fácil de usar):**
```python
def busqueda_binaria(arr, objetivo):
    \"\"\"Wrapper que oculta los parámetros izq/der\"\"\"
    return busqueda_binaria_recursiva(arr, objetivo, 0, len(arr) - 1)

# Uso simplificado
print(busqueda_binaria([1, 2, 3, 4, 5], 4))  # Output: 3
```

**Comparación: Recursiva vs Iterativa**

| Aspecto | Recursiva | Iterativa |
|---------|-----------|-----------|
| Legibilidad | Más intuitiva | Menos clara |
| Espacio | O(log n) pila | O(1) |
| Velocidad | Ligeramente más lenta | Más rápida |
| Límite | Stack overflow posible | Sin límite |

**Cuándo usar recursiva:**
- Código educativo/académico
- Listas pequeñas-medianas
- Cuando la claridad es prioritaria

**Cuándo usar iterativa:**
- Producción/performance crítica
- Listas muy grandes
- Sistemas con stack limitado

```python
# Ambas tienen la misma complejidad
# Tiempo: O(log n)
# Recursiva espacio: O(log n) - pila de llamadas
# Iterativa espacio: O(1)
```

¡La recursiva es más elegante, pero la iterativa es más eficiente! 🔄🔍"""
    })

    return ejemplos


def generar_estructuras_datos():
    """Genera ejemplos sobre estructuras de datos"""
    ejemplos = []

    # Listas enlazadas
    ejemplos.append({
        "instruction": "¿Qué es una lista enlazada y cómo implementarla en Python?",
        "input": "",
        "output": """Claro viejito, una lista enlazada es como un tren: cada vagón (nodo) tiene su carga (dato) y sabe cuál es el siguiente vagón. A diferencia de las listas Python normales, no están en memoria contigua.

**Estructura básica:**
```python
class Nodo:
    \"\"\"Representa un vagón del tren\"\"\"
    def __init__(self, dato):
        self.dato = dato
        self.siguiente = None  # Apunta al próximo vagón

class ListaEnlazada:
    \"\"\"El tren completo\"\"\"
    def __init__(self):
        self.cabeza = None  # Primer vagón

    def esta_vacia(self):
        return self.cabeza is None

    def agregar_al_inicio(self, dato):
        \"\"\"Agrega un vagón al frente del tren\"\"\"
        nuevo_nodo = Nodo(dato)
        nuevo_nodo.siguiente = self.cabeza
        self.cabeza = nuevo_nodo

    def agregar_al_final(self, dato):
        \"\"\"Agrega un vagón al final del tren\"\"\"
        nuevo_nodo = Nodo(dato)

        # Si el tren está vacío
        if self.esta_vacia():
            self.cabeza = nuevo_nodo
            return

        # Recorrer hasta el último vagón
        actual = self.cabeza
        while actual.siguiente:
            actual = actual.siguiente

        actual.siguiente = nuevo_nodo

    def eliminar(self, dato):
        \"\"\"Desconecta un vagón del tren\"\"\"
        if self.esta_vacia():
            return False

        # Si es el primer vagón
        if self.cabeza.dato == dato:
            self.cabeza = self.cabeza.siguiente
            return True

        # Buscar el vagón
        actual = self.cabeza
        while actual.siguiente:
            if actual.siguiente.dato == dato:
                actual.siguiente = actual.siguiente.siguiente
                return True
            actual = actual.siguiente

        return False

    def buscar(self, dato):
        \"\"\"Busca un vagón con cierto dato\"\"\"
        actual = self.cabeza
        posicion = 0

        while actual:
            if actual.dato == dato:
                return posicion
            actual = actual.siguiente
            posicion += 1

        return -1

    def imprimir(self):
        \"\"\"Muestra todos los vagones del tren\"\"\"
        if self.esta_vacia():
            print(\"Lista vacía\")
            return

        actual = self.cabeza
        elementos = []
        while actual:
            elementos.append(str(actual.dato))
            actual = actual.siguiente

        print(\" -> \".join(elementos))

    def longitud(self):
        \"\"\"Cuenta cuántos vagones tiene el tren\"\"\"
        count = 0
        actual = self.cabeza
        while actual:
            count += 1
            actual = actual.siguiente
        return count

# Ejemplo de uso
tren = ListaEnlazada()

# Agregar vagones
tren.agregar_al_final(10)
tren.agregar_al_final(20)
tren.agregar_al_final(30)
tren.agregar_al_inicio(5)

tren.imprimir()  # Output: 5 -> 10 -> 20 -> 30

# Buscar
print(f\"20 está en posición: {tren.buscar(20)}\")  # Output: 2

# Eliminar
tren.eliminar(20)
tren.imprimir()  # Output: 5 -> 10 -> 30

print(f\"Longitud: {tren.longitud()}\")  # Output: 3
```

**Ventajas vs listas Python:**
- Inserción/eliminación al inicio: O(1) vs O(n)
- No necesita memoria contigua
- Tamaño dinámico sin redimensionar

**Desventajas:**
- Acceso por índice: O(n) vs O(1)
- Más memoria por nodo (guarda siguiente)
- No se puede acceder directamente a un elemento

**Complejidades:**

| Operación | Lista Enlazada | Lista Python |
|-----------|---------------|--------------|
| Agregar inicio | O(1) | O(n) |
| Agregar final | O(n) | O(1) amortizado |
| Eliminar inicio | O(1) | O(n) |
| Acceso por índice | O(n) | O(1) |
| Buscar | O(n) | O(n) |

**Cuándo usar lista enlazada:**
- Muchas inserciones/eliminaciones al inicio
- Tamaño muy variable
- Implementar pilas o colas

**Cuándo NO usarla:**
- Necesitas acceso aleatorio frecuente
- Memoria limitada
- Simplicidad es prioritaria (usa list de Python)

¡Las listas enlazadas son como trenes: fáciles de conectar/desconectar vagones, pero lentas para llegar al vagón N! 🚂🔗"""
    })

    return ejemplos


def generar_ordenamiento():
    """Genera ejemplos sobre algoritmos de ordenamiento"""
    ejemplos = []

    # Bubble sort
    ejemplos.append({
        "instruction": "Explícame bubble sort con código y visualización",
        "input": "",
        "output": """Claro viejito, bubble sort es como burbujas que suben: los elementos más grandes "burbujean" hacia el final comparando e intercambiando pares adyacentes.

**Concepto:**
Compara elementos vecinos y los intercambia si están en orden incorrecto. Repite hasta que no haya más intercambios.

**Implementación paso a paso:**
```python
def bubble_sort(arr):
    \"\"\"
    Ordena una lista usando bubble sort.
    Complejidad: O(n²)
    \"\"\"
    n = len(arr)

    # Necesitamos n-1 pasadas
    for i in range(n - 1):
        # Bandera para detectar si hubo intercambios
        hubo_intercambio = False

        # Última i elementos ya están ordenados
        for j in range(n - 1 - i):
            # Comparar elementos adyacentes
            if arr[j] > arr[j + 1]:
                # Intercambiar
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                hubo_intercambio = True

        # Si no hubo intercambios, ya está ordenado
        if not hubo_intercambio:
            break

    return arr

# Ejemplo
numeros = [64, 34, 25, 12, 22, 11, 90]
print(f\"Original: {numeros}\")
bubble_sort(numeros)
print(f\"Ordenado: {numeros}\")
```

**Visualización paso a paso:**
```
Array inicial: [64, 34, 25, 12, 22, 11, 90]

Pasada 1:
[64, 34, 25, 12, 22, 11, 90]  Comparar 64 y 34 → intercambiar
[34, 64, 25, 12, 22, 11, 90]  Comparar 64 y 25 → intercambiar
[34, 25, 64, 12, 22, 11, 90]  Comparar 64 y 12 → intercambiar
[34, 25, 12, 64, 22, 11, 90]  Comparar 64 y 22 → intercambiar
[34, 25, 12, 22, 64, 11, 90]  Comparar 64 y 11 → intercambiar
[34, 25, 12, 22, 11, 64, 90]  Comparar 64 y 90 → NO intercambiar
[34, 25, 12, 22, 11, 64, 90]  ← 90 ya está en su lugar

Pasada 2:
[25, 34, 12, 22, 11, 64, 90]  Después de burbujear...
[25, 12, 34, 22, 11, 64, 90]
[25, 12, 22, 34, 11, 64, 90]
[25, 12, 22, 11, 34, 64, 90]  ← 64 y 90 en su lugar

... continúa hasta ordenar todo
```

**Versión con visualización:**
```python
def bubble_sort_verbose(arr):
    \"\"\"Bubble sort que muestra cada paso\"\"\"
    n = len(arr)
    print(f\"Inicial: {arr}\\n\")

    for i in range(n - 1):
        print(f\"--- Pasada {i + 1} ---\")
        hubo_intercambio = False

        for j in range(n - 1 - i):
            print(f\"Comparar {arr[j]} y {arr[j+1]}\", end=\" \")

            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                hubo_intercambio = True
                print(f\"→ Intercambiar: {arr}\")
            else:
                print(\"→ No intercambiar\")

        if not hubo_intercambio:
            print(\"No hubo intercambios. ¡Ya está ordenado!\\n\")
            break
        print()

    return arr

# Prueba
numeros = [5, 2, 8, 1, 9]
bubble_sort_verbose(numeros)
```

**Optimizaciones:**

1. **Detección temprana (ya implementada):**
```python
if not hubo_intercambio:
    break  # Si pasada sin intercambios = ordenado
```

2. **Reducir rango en cada pasada:**
```python
for j in range(n - 1 - i):  # Últimos i ya ordenados
```

**Análisis de complejidad:**

| Caso | Complejidad | Ejemplo |
|------|-------------|---------|
| Mejor | O(n) | Lista ya ordenada [1,2,3,4,5] |
| Promedio | O(n²) | Lista aleatoria [3,1,4,2,5] |
| Peor | O(n²) | Lista inversa [5,4,3,2,1] |

**Espacio:** O(1) - ordena in-place

**Ventajas:**
- Simple de entender e implementar
- Estable (mantiene orden relativo de elementos iguales)
- In-place (no usa memoria extra)

**Desventajas:**
- Muy lento para listas grandes
- O(n²) es ineficiente
- Muchas comparaciones e intercambios

**Cuándo usarlo:**
- Listas pequeñas (< 10 elementos)
- Propósito educativo
- Cuando simplicidad > eficiencia

**Comparación con otros algoritmos:**
```python
import time
import random

# Generar lista aleatoria
lista = [random.randint(1, 1000) for _ in range(1000)]

# Bubble sort
lista_bubble = lista.copy()
start = time.time()
bubble_sort(lista_bubble)
time_bubble = time.time() - start

# Python sorted (Timsort - O(n log n))
lista_python = lista.copy()
start = time.time()
lista_python.sort()
time_python = time.time() - start

print(f\"Bubble Sort: {time_bubble:.4f}s\")
print(f\"Python Sort: {time_python:.4f}s\")
print(f\"Python es {time_bubble/time_python:.0f}x más rápido\")
```

¡Bubble sort es fácil de entender pero lento - como ordenar barajando cartas comparando solo pares vecinos! 🫧📊"""
    })

    return ejemplos


# Función principal de expansión
def expandir_dataset():
    """Expande el dataset generando muchos más ejemplos"""

    print("[INFO] Generando ejemplos adicionales...")

    todos_ejemplos = []

    # Generar ejemplos por categoría
    print("  - Algoritmos de búsqueda...")
    todos_ejemplos.extend(generar_algoritmos_busqueda())

    print("  - Estructuras de datos...")
    todos_ejemplos.extend(generar_estructuras_datos())

    print("  - Algoritmos de ordenamiento...")
    todos_ejemplos.extend(generar_ordenamiento())

    print(f"\\n[SUCCESS] Generados {len(todos_ejemplos)} ejemplos nuevos")

    return todos_ejemplos


if __name__ == "__main__":
    ejemplos = expandir_dataset()

    # Guardar en archivo temporal
    output_file = "data/ejemplos_adicionales_enriquecidos.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ejemplos, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Guardados en: {output_file}")
