import numpy as np
import cv2 as cv
import os
from datetime import datetime

def capturar_automatico_opencv():
    """
    Captura automática de 100 fotos usando SOLO OpenCV
    Sin TensorFlow ni librerías complejas
    """
    
    CARPETA_FOTOS = "dataset_rostros"
    
    print("\n" + "="*70)
    print("  📸 CAPTURA AUTOMÁTICA - SOLO OpenCV")
    print("="*70)
    
    NOMBRE_PERSONA = input("\n📝 Ingrese el nombre de la persona: ").strip()
    
    if not NOMBRE_PERSONA:
        print("❌ Debe ingresar un nombre válido")
        return
    
    ruta_persona = os.path.join(CARPETA_FOTOS, NOMBRE_PERSONA)
    os.makedirs(ruta_persona, exist_ok=True)
    
    print(f"\n✅ Las fotos se guardarán en: {ruta_persona}")
    
    # Cargar clasificador
    rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
    if rostro.empty():
        print("❌ ERROR: No se encontró haarcascade_frontalface_alt.xml")
        return
    
    # Preguntar fuente de video
    print("\n🎥 Selecciona la fuente:")
    print("  1 - Cámara web (0)")
    print("  2 - Archivo de video")
    opcion = input("\nOpción: ").strip()
    
    if opcion == "2":
        ruta_video = input("📁 Ruta del video: ").strip()
        cap = cv.VideoCapture(ruta_video)
    else:
        cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: No se pudo abrir la fuente de video")
        return
    
    print("\n" + "="*70)
    print("📸 MODO DE CAPTURA AUTOMÁTICA")
    print("="*70)
    print("\n⚡ INSTRUCCIONES:")
    print("  • Presiona ESPACIO para INICIAR/PAUSAR captura automática")
    print("  • Captura cada 10 frames automáticamente")
    print("  • Mueve tu cabeza lentamente")
    print("  • Cambia expresiones")
    print("  • Presiona ESC para terminar")
    print(f"  • Objetivo: 100 fotos (se convertirán en 3000)")
    print("="*70 + "\n")
    
    contador = 0
    frame_count = 0
    capturando = False
    FOTOS_OBJETIVO = 100
    
    print("⏸️  Presiona ESPACIO para comenzar\n")
    
    while contador < FOTOS_OBJETIVO:
        ret, frame = cap.read()
        
        if not ret:
            print("\n⚠️  Fin del video o error en la cámara")
            break
        
        frame_count += 1
        
        # Detectar rostros
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rostros = rostro.detectMultiScale(gray, 1.3, 5)
        
        # Dibujar rostros
        for (x, y, w, h) in rostros:
            color = (0, 255, 0) if capturando else (0, 165, 255)
            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            estado = "CAPTURANDO" if capturando else "PAUSADO"
            cv.putText(frame, estado, (x, y-10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Información en pantalla
        texto_progreso = f"Fotos: {contador}/{FOTOS_OBJETIVO}"
        cv.putText(frame, texto_progreso, (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        imagenes_totales = contador * 30
        texto_total = f"Total con augmentation: {imagenes_totales}"
        cv.putText(frame, texto_total, (10, 70),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        estado_texto = "PAUSADO (ESPACIO: Iniciar)" if not capturando else "CAPTURANDO"
        cv.putText(frame, estado_texto, (10, frame.shape[0] - 20),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Captura automática cada 10 frames
        if capturando and frame_count % 10 == 0:
            if len(rostros) > 0:
                # Tomar primer rostro
                (x, y, w, h) = rostros[0]
                rostro_recortado = frame[y:y+h, x:x+w]
                
                # Guardar
                nombre_archivo = f"{NOMBRE_PERSONA}_{contador+1:04d}.jpg"
                ruta_completa = os.path.join(ruta_persona, nombre_archivo)
                cv.imwrite(ruta_completa, rostro_recortado)
                
                contador += 1
                
                if contador % 10 == 0:
                    print(f"✅ {contador}/{FOTOS_OBJETIVO} fotos capturadas")
        
        cv.imshow('Captura Automatica - OpenCV', frame)
        
        k = cv.waitKey(1)
        if k == 27:  # ESC
            break
        elif k == 32:  # ESPACIO
            capturando = not capturando
            estado = "INICIANDO" if capturando else "PAUSADO"
            print(f"\n{'▶️' if capturando else '⏸️'} {estado}")
    
    cap.release()
    cv.destroyAllWindows()
    
    imagenes_totales = contador * 30
    
    print("\n" + "="*70)
    print("📊 RESUMEN DE CAPTURA")
    print("="*70)
    print(f"  👤 Persona: {NOMBRE_PERSONA}")
    print(f"  📸 Fotos capturadas: {contador}")
    print(f"  🎨 Se generarán: {imagenes_totales} imágenes con augmentation")
    print(f"  📁 Ubicación: {ruta_persona}")
    print("="*70)
    
    if contador >= 100:
        print(f"\n✅ ¡Perfecto! Ahora ejecuta:")
        print(f"   python augmentation_opencv.py")
    else:
        print(f"\n⚠️  Solo capturaste {contador} fotos")
        print(f"   Tendrás ~{imagenes_totales} imágenes con augmentation")

if __name__ == "__main__":
    capturar_automatico_opencv()
