import os
import random
from PIL import Image, ImageEnhance, ImageOps

# --- CONFIGURACIÓN ---
# Detecta la ruta automáticamente si estás en Unidad2
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset")

META_IMAGENES = 10000  # Queremos llegar a este número por clase

def transformar_imagen(img):
    """Aplica una transformación aleatoria para crear una imagen 'nueva'"""
    # 1. Espejo (50% probabilidad)
    if random.choice([True, False]):
        img = ImageOps.mirror(img)
    
    # 2. Rotación leve (entre -20 y 20 grados)
    angulo = random.randint(-20, 20)
    img = img.rotate(angulo, resample=Image.Resampling.BICUBIC, expand=False)
    
    # 3. Color/Brillo (variación sutil)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2)) # 80% a 120% de brillo
    
    return img

def balancear_clases():
    if not os.path.exists(dataset_dir):
        print("❌ No encuentro la carpeta 'dataset'.")
        return

    clases = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print(f"🎯 Meta: {META_IMAGENES} imágenes por clase.\n")

    for clase in clases:
        ruta_clase = os.path.join(dataset_dir, clase)
        imagenes = [f for f in os.listdir(ruta_clase) if f.lower().endswith(('.jpg', '.png'))]
        cantidad_actual = len(imagenes)
        
        print(f"📂 Clase '{clase}': {cantidad_actual} imágenes.")
        
        if cantidad_actual >= META_IMAGENES:
            print(f"   ✅ Ya está completa. Saltando...")
            continue
        
        faltantes = META_IMAGENES - cantidad_actual
        print(f"   ⚠️ Faltan {faltantes}. Generando clones...")
        
        # Generar imágenes nuevas
        contador = 0
        while contador < faltantes:
            # Elegir una imagen original al azar para clonar
            imagen_azar = random.choice(imagenes)
            ruta_orig = os.path.join(ruta_clase, imagen_azar)
            
            try:
                with Image.open(ruta_orig) as img:
                    # Convertir y transformar
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img_nueva = transformar_imagen(img)
                    
                    # Guardar con nombre único
                    nombre_nuevo = f"aug_{contador}_{imagen_azar}"
                    img_nueva.save(os.path.join(ruta_clase, nombre_nuevo), quality=90)
                    
                    contador += 1
                    
                    if contador % 500 == 0:
                        print(f"      -> Generadas {contador}/{faltantes}...")
                        
            except Exception as e:
                print(f"Error con {imagen_azar}: {e}")
                continue
                
        print(f"   ✨ ¡Listo! Ahora '{clase}' tiene {len(os.listdir(ruta_clase))} imágenes.\n")

if __name__ == "__main__":
    balancear_clases()
    print("🎉 Dataset balanceado correctamente.")