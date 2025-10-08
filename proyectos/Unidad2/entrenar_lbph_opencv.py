import numpy as np
import cv2 as cv
import os

def entrenar_lbph_opencv():
    """
    Entrena modelo LBPH usando SOLO OpenCV
    Sin TensorFlow ni scikit-learn
    """
    
    print("\n" + "="*70)
    print("  🧠 ENTRENAMIENTO LBPH - SOLO OpenCV")
    print("="*70 + "\n")
    
    CARPETA_DATASET = "dataset_rostros_augmented"
    
    if not os.path.exists(CARPETA_DATASET):
        print(f"❌ ERROR: No existe '{CARPETA_DATASET}'")
        print("\n📋 Pasos necesarios:")
        print("   1. python captura_opencv.py")
        print("   2. python augmentation_opencv.py")
        print("   3. python entrenar_lbph_opencv.py (este)")
        return
    
    # Obtener personas
    personas = [p for p in os.listdir(CARPETA_DATASET) 
                if os.path.isdir(os.path.join(CARPETA_DATASET, p))]
    
    if len(personas) == 0:
        print("❌ No hay personas en el dataset")
        return
    
    print(f"👥 Personas encontradas: {len(personas)}")
    for i, persona in enumerate(personas):
        print(f"   {i} - {persona}")
    
    # Cargar imágenes
    print("\n📂 Cargando imágenes...")
    
    rostros = []
    etiquetas = []
    nombres = {}  # Mapeo de ID a nombre
    
    total_imagenes = 0
    
    for id_persona, nombre_persona in enumerate(personas):
        nombres[id_persona] = nombre_persona
        ruta_persona = os.path.join(CARPETA_DATASET, nombre_persona)
        
        archivos = [f for f in os.listdir(ruta_persona) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   📁 {nombre_persona}: {len(archivos)} imágenes")
        
        for archivo in archivos:
            ruta_img = os.path.join(ruta_persona, archivo)
            img = cv.imread(ruta_img, cv.IMREAD_GRAYSCALE)
            
            if img is not None:
                rostros.append(img)
                etiquetas.append(id_persona)
                total_imagenes += 1
    
    print(f"\n✅ Total cargado: {total_imagenes} imágenes")
    print(f"👤 Personas: {len(personas)}")
    
    if total_imagenes < 100:
        print("\n⚠️  Tienes pocas imágenes. Recomendado: 1000+")
    
    # Crear reconocedor LBPH
    print("\n🔧 Creando y entrenando modelo LBPH...")
    
    reconocedor = cv.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    
    # Entrenar
    reconocedor.train(rostros, np.array(etiquetas))
    
    print("✅ Entrenamiento completado")
    
    # Guardar modelo
    reconocedor.save('modelo_lbph.yml')
    print("💾 Modelo guardado: modelo_lbph.yml")
    
    # Guardar nombres en archivo de texto
    with open('nombres.txt', 'w', encoding='utf-8') as f:
        for id_persona, nombre in nombres.items():
            f.write(f"{id_persona}:{nombre}\n")
    print("💾 Nombres guardados: nombres.txt")
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"  📊 Imágenes entrenadas: {total_imagenes}")
    print(f"  👥 Personas: {len(personas)}")
    print(f"  📄 Archivos generados:")
    print(f"     • modelo_lbph.yml")
    print(f"     • nombres.txt")
    print("="*70)
    print(f"\n📋 SIGUIENTE PASO:")
    print(f"   python reconocer_lbph_opencv.py")

if __name__ == "__main__":
    try:
        entrenar_lbph_opencv()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n💡 Asegúrate de tener instalado opencv-contrib-python:")
        print("   pip install opencv-contrib-python")
