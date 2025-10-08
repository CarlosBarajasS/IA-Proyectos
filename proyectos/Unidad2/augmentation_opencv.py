import numpy as np
import cv2 as cv
import os

def augmentation_opencv():
    """
    Data augmentation usando SOLO OpenCV y NumPy
    Genera 30 versiones de cada foto: 100 fotos -> 3000 imágenes
    """
    
    CARPETA_ORIGEN = "dataset_rostros"
    CARPETA_DESTINO = "dataset_rostros_augmented"
    VERSIONES = 30
    
    print("\n" + "="*70)
    print("  🎨 DATA AUGMENTATION - SOLO OpenCV")
    print("  📊 Generando 3000+ imágenes")
    print("="*70 + "\n")
    
    if not os.path.exists(CARPETA_ORIGEN):
        print(f"❌ ERROR: No existe '{CARPETA_ORIGEN}'")
        print("   Primero ejecuta: python captura_opencv.py")
        return
    
    # Obtener personas
    personas = [p for p in os.listdir(CARPETA_ORIGEN) 
                if os.path.isdir(os.path.join(CARPETA_ORIGEN, p))]
    
    if len(personas) == 0:
        print("❌ No hay personas en el dataset")
        return
    
    print(f"👥 Personas encontradas: {len(personas)}")
    for persona in personas:
        print(f"   • {persona}")
    
    print(f"\n🎨 Generando {VERSIONES} versiones por foto...\n")
    
    total_generadas = 0
    
    for persona in personas:
        ruta_origen = os.path.join(CARPETA_ORIGEN, persona)
        ruta_destino = os.path.join(CARPETA_DESTINO, persona)
        os.makedirs(ruta_destino, exist_ok=True)
        
        fotos = [f for f in os.listdir(ruta_origen) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📁 {persona}: {len(fotos)} fotos originales")
        
        contador = 0
        for foto in fotos:
            ruta_foto = os.path.join(ruta_origen, foto)
            img = cv.imread(ruta_foto)
            
            if img is None:
                continue
            
            # Generar versiones
            versiones = generar_versiones_opencv(img)
            
            # Guardar cada versión
            nombre_base = os.path.splitext(foto)[0]
            for idx, version in enumerate(versiones):
                nombre = f"{nombre_base}_aug{idx+1:02d}.jpg"
                ruta_completa = os.path.join(ruta_destino, nombre)
                cv.imwrite(ruta_completa, version)
                contador += 1
                total_generadas += 1
        
        print(f"   ✅ Generadas: {contador} imágenes\n")
    
    print("="*70)
    print("✅ AUGMENTATION COMPLETADO")
    print("="*70)
    print(f"  📊 Total generado: {total_generadas} imágenes")
    print(f"  📁 Ubicación: {CARPETA_DESTINO}/")
    print("="*70)
    print(f"\n📋 SIGUIENTE PASO:")
    print(f"   python entrenar_lbph_opencv.py")

def generar_versiones_opencv(img):
    """
    Genera 30 versiones de una imagen usando SOLO OpenCV
    """
    versiones = []
    h, w = img.shape[:2]
    
    # 1. Original
    versiones.append(img.copy())
    
    # 2. Flip horizontal
    versiones.append(cv.flip(img, 1))
    
    # 3-7. Rotaciones
    for angulo in [-15, -8, 8, 15, 20]:
        M = cv.getRotationMatrix2D((w/2, h/2), angulo, 1.0)
        rotada = cv.warpAffine(img, M, (w, h))
        versiones.append(rotada)
    
    # 8-12. Brillo
    for beta in [-50, -30, 30, 50, 70]:
        brillo = cv.convertScaleAbs(img, alpha=1.0, beta=beta)
        versiones.append(brillo)
    
    # 13-16. Contraste
    for alpha in [0.7, 0.85, 1.15, 1.3]:
        contraste = cv.convertScaleAbs(img, alpha=alpha, beta=0)
        versiones.append(contraste)
    
    # 17-19. Blur
    for kernel in [3, 5, 7]:
        blur = cv.GaussianBlur(img, (kernel, kernel), 0)
        versiones.append(blur)
    
    # 20-22. Ruido
    for intensidad in [10, 20, 25]:
        ruido = img.copy().astype(np.int16)
        noise = np.random.randint(-intensidad, intensidad, img.shape, dtype=np.int16)
        con_ruido = np.clip(ruido + noise, 0, 255).astype(np.uint8)
        versiones.append(con_ruido)
    
    # 23-25. Desplazamiento
    for dx, dy in [(5, 5), (-5, 5), (5, -5)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        desplazada = cv.warpAffine(img, M, (w, h))
        versiones.append(desplazada)
    
    # 26-28. Zoom
    for factor in [0.9, 1.1, 1.15]:
        M = cv.getRotationMatrix2D((w/2, h/2), 0, factor)
        zoom = cv.warpAffine(img, M, (w, h))
        versiones.append(zoom)
    
    # 29. Flip vertical
    versiones.append(cv.flip(img, 0))
    
    # 30. Combinación: rotación + brillo
    M = cv.getRotationMatrix2D((w/2, h/2), 10, 1.0)
    temp = cv.warpAffine(img, M, (w, h))
    combo = cv.convertScaleAbs(temp, alpha=1.2, beta=20)
    versiones.append(combo)
    
    return versiones[:30]  # Exactamente 30 versiones

if __name__ == "__main__":
    augmentation_opencv()
