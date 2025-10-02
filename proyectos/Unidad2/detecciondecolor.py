import cv2
import numpy as np
import os

def detectar_por_color(imagen_path):
    # Verificar si el archivo existe
    if not os.path.exists(imagen_path):
        print(f"\n❌ ERROR: No se encontró el archivo en la ruta: {imagen_path}")
        print(f"📁 Directorio actual: {os.getcwd()}")
        return
    
    # Leer la imagen
    imagen = cv2.imread(imagen_path)
    
    if imagen is None:
        print(f"\n❌ ERROR: No se pudo cargar la imagen: {imagen_path}")
        return
    
    print(f"✅ Imagen cargada correctamente: {imagen.shape}")
    
    imagen_original = imagen.copy()
    
    # Convertir a HSV para mejor detección de colores
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # Definir rangos de colores en HSV
    colores = {
        '1': {
            'nombre': 'Rojo', 
            'lower': np.array([0, 120, 70]), 
            'upper': np.array([10, 255, 255]),
            'color_marca': (0, 0, 255)  # Rojo en BGR
        },
        '2': {
            'nombre': 'Verde', 
            'lower': np.array([40, 50, 50]), 
            'upper': np.array([80, 255, 255]),
            'color_marca': (0, 255, 0)  # Verde en BGR
        },
        '3': {
            'nombre': 'Azul', 
            'lower': np.array([100, 150, 50]), 
            'upper': np.array([130, 255, 255]),
            'color_marca': (255, 0, 0)  # Azul en BGR
        },
        '4': {
            'nombre': 'Amarillo', 
            'lower': np.array([20, 100, 100]), 
            'upper': np.array([30, 255, 255]),
            'color_marca': (0, 255, 255)  # Amarillo en BGR
        }
    }
    
    # Mostrar menú
    print("\n=== DETECTOR DE FORMAS POR COLOR ===")
    print("Seleccione el color a buscar:")
    print("1 - Rojo")
    print("2 - Verde")
    print("3 - Azul")
    print("4 - Amarillo")
    print("0 - Salir")
    
    opcion = input("\nIngrese el número del color: ")
    
    if opcion == '0':
        return
    
    if opcion not in colores:
        print("❌ Opción no válida")
        return
    
    color_seleccionado = colores[opcion]
    print(f"\n🔍 Buscando formas de color: {color_seleccionado['nombre']}")
    
    # Crear máscara para el color seleccionado
    mascara = cv2.inRange(hsv, color_seleccionado['lower'], color_seleccionado['upper'])
    
    # Para el rojo, agregar el rango superior
    if opcion == '1':
        mascara2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mascara = cv2.bitwise_or(mascara, mascara2)
    
    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    
    # Crear una imagen de resultado
    resultado = imagen_original.copy()
    
    # Marcar las zonas detectadas con el color correspondiente (semitransparente)
    color_overlay = np.zeros_like(imagen_original)
    color_overlay[mascara > 0] = color_seleccionado['color_marca']
    resultado = cv2.addWeighted(imagen_original, 0.7, color_overlay, 0.3, 0)
    
    # Encontrar contornos solo para calcular centros
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n📍 COORDENADAS DE LOS CENTROS:")
    print("=" * 50)
    
    contador = 0
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500:  # Filtrar contornos pequeños
            # Calcular momentos para encontrar el centro
            M = cv2.moments(contorno)
            
            if M["m00"] != 0:
                # Calcular coordenadas del centro
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                contador += 1
                
                # Imprimir coordenadas en consola
                print(f"Forma #{contador}: Centro en (X={cx}, Y={cy})")
                
                # Dibujar un círculo en el centro
                cv2.circle(resultado, (cx, cy), 8, (255, 255, 255), -1)  # Círculo blanco relleno
                cv2.circle(resultado, (cx, cy), 10, (0, 0, 0), 2)  # Borde negro
                
                # Mostrar las coordenadas en la imagen
                texto = f"#{contador} ({cx},{cy})"
                cv2.putText(resultado, texto, (cx - 50, cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(resultado, texto, (cx - 50, cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    print("=" * 50)
    print(f"\n✅ Total de formas detectadas: {contador}")
    
    # Contar píxeles detectados
    pixeles_detectados = np.count_nonzero(mascara)
    porcentaje = (pixeles_detectados / (mascara.shape[0] * mascara.shape[1])) * 100
    print(f"📊 Píxeles detectados: {pixeles_detectados}")
    print(f"📊 Porcentaje de la imagen: {porcentaje:.2f}%")
    
    # Mostrar resultados
    cv2.imshow('1. Imagen Original', imagen_original)
    cv2.imshow('2. Mascara de Color', mascara)
    cv2.imshow(f'3. Deteccion con Centros - {color_seleccionado["nombre"]}', resultado)
    
    print("\n⌨️  Presione cualquier tecla en las ventanas para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uso del programa
if __name__ == "__main__":
    print(f"📁 Directorio de trabajo actual: {os.getcwd()}\n")
    
    # Cambia el nombre de tu archivo aquí
    ruta_imagen = "image.png"
    
    detectar_por_color(ruta_imagen)