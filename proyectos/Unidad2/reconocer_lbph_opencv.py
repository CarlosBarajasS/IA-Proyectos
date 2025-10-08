import numpy as np
import cv2 as cv
import os

def reconocer_lbph_opencv():
    """
    Reconocimiento facial en tiempo real usando LBPH de OpenCV
    Sin TensorFlow ni otras librerías complejas
    """
    
    print("\n" + "="*70)
    print("  🎥 RECONOCIMIENTO FACIAL - LBPH OpenCV")
    print("="*70 + "\n")
    
    # Verificar archivos
    if not os.path.exists('modelo_lbph.yml'):
        print("❌ ERROR: No se encontró modelo_lbph.yml")
        print("   Ejecuta primero: python entrenar_lbph_opencv.py")
        return
    
    if not os.path.exists('nombres.txt'):
        print("❌ ERROR: No se encontró nombres.txt")
        print("   Ejecuta primero: python entrenar_lbph_opencv.py")
        return
    
    if not os.path.exists('haarcascade_frontalface_alt.xml'):
        print("❌ ERROR: No se encontró haarcascade_frontalface_alt.xml")
        return
    
    # Cargar modelo
    print("📥 Cargando modelo LBPH...")
    reconocedor = cv.face.LBPHFaceRecognizer_create()
    reconocedor.read('modelo_lbph.yml')
    print("   ✅ Modelo cargado")
    
    # Cargar nombres
    nombres = {}
    with open('nombres.txt', 'r', encoding='utf-8') as f:
        for linea in f:
            id_persona, nombre = linea.strip().split(':')
            nombres[int(id_persona)] = nombre
    
    print(f"   ✅ {len(nombres)} personas cargadas")
    for id_persona, nombre in nombres.items():
        print(f"      • {nombre}")
    
    # Cargar clasificador de rostros
    rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
    if rostro.empty():
        print("❌ ERROR: No se pudo cargar haarcascade_frontalface_alt.xml")
        return
    
    print("\n" + "="*70)
    print("📸 INSTRUCCIONES:")
    print("  • El sistema reconocerá rostros automáticamente")
    print("  • Verde = Alta confianza (<50)")
    print("  • Amarillo = Media confianza (50-70)")
    print("  • Rojo = Baja confianza (>70)")
    print("  • Presiona Q o ESC para salir")
    print("="*70 + "\n")
    
    # Seleccionar fuente
    print("🎥 Selecciona la fuente:")
    print("  1 - Cámara web")
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
    
    print("\n🎥 Reconociendo rostros...")
    print("   Presiona Q o ESC para salir\n")
    
    # Estadísticas
    stats = {nombre: 0 for nombre in nombres.values()}
    stats['Desconocido'] = 0
    frames = 0
    
    # Umbral de confianza (menor es mejor en LBPH)
    UMBRAL = 70
    
    while True:
        ret, img = cap.read()
        
        if not ret:
            print("\n⚠️  Fin del video")
            break
        
        frames += 1
        
        # Convertir a gris
        gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Detectar rostros
        rostros = rostro.detectMultiScale(gris, 1.3, 5)
        
        # Procesar cada rostro
        for (x, y, w, h) in rostros:
            # Extraer región del rostro
            rostro_gris = gris[y:y+h, x:x+w]
            
            try:
                # Predecir
                id_predicho, confianza = reconocedor.predict(rostro_gris)
                
                # En LBPH, menor confianza = mejor
                if confianza < UMBRAL:
                    nombre = nombres[id_predicho]
                    porcentaje = 100 - confianza
                    stats[nombre] += 1
                    
                    # Color según confianza
                    if confianza < 40:
                        color = (0, 255, 0)  # Verde
                        estado = "Alta"
                    elif confianza < 60:
                        color = (0, 200, 200)  # Amarillo
                        estado = "Media"
                    else:
                        color = (0, 165, 255)  # Naranja
                        estado = "Baja"
                else:
                    nombre = "Desconocido"
                    porcentaje = 0
                    color = (0, 0, 255)  # Rojo
                    estado = "Muy baja"
                    stats['Desconocido'] += 1
                
                # Dibujar rectángulo
                cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Fondo para texto
                cv.rectangle(img, (x, y-70), (x+w, y), color, -1)
                
                # Nombre
                cv.putText(img, nombre, (x+5, y-45),
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Confianza
                texto_conf = f"Conf: {confianza:.0f}"
                cv.putText(img, texto_conf, (x+5, y-20),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Estado
                cv.putText(img, estado, (x+5, y-5),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                # Rostro sin clasificar
                cv.rectangle(img, (x, y), (x+w, y+h), (128, 128, 128), 2)
        
        # Información en pantalla
        texto_rostros = f"Rostros: {len(rostros)}"
        cv.putText(img, texto_rostros, (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(img, texto_rostros, (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        cv.putText(img, "Q o ESC: Salir", (10, img.shape[0]-20),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv.imshow('Reconocimiento Facial - LBPH', img)
        
        k = cv.waitKey(1)
        if k == 27 or k == ord('q'):  # ESC o Q
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    # Mostrar estadísticas
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS")
    print("="*70)
    print(f"  🎬 Frames procesados: {frames}")
    print(f"\n  👥 Detecciones:")
    
    total = sum(stats.values())
    for nombre, cantidad in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        if cantidad > 0:
            porcentaje = (cantidad / total * 100) if total > 0 else 0
            print(f"     • {nombre}: {cantidad} ({porcentaje:.1f}%)")
    
    print("="*70)
    print("👋 ¡Hasta pronto!\n")

if __name__ == "__main__":
    try:
        reconocer_lbph_opencv()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programa interrumpido")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
