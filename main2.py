import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import graycomatrix, graycoprops

# Función para calcular las características GLCM (corregida)
def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    # Verificar si la imagen es uniforme
    if np.std(image) == 0:
        raise ValueError("Imagen uniforme: no se puede calcular GLCM")
    
    # Calcular la matriz GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    # Calcular propiedades GLCM (manejo de NaN en correlación)
    try:
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        asm = graycoprops(glcm, 'ASM').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = np.nan_to_num(graycoprops(glcm, 'correlation').mean())  # Manejar NaN
        variance = graycoprops(glcm, 'variance').mean()
        entropy = graycoprops(glcm, 'entropy').mean()
    except Exception as e:
        raise ValueError(f"Error en propiedades GLCM: {e}")
    
    return np.array([contrast, dissimilarity, homogeneity, asm, energy, correlation, variance, entropy])

# Función para preprocesar la imagen (sin cambios)
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error al cargar: {image_path}")
    
    # Ecualización y binarización
    equalized_img = cv2.equalizeHist(img)
    _, binary_img = cv2.threshold(equalized_img, 228, 255, cv2.THRESH_BINARY)

    # Comparativas umbralizacion global
    """ _, binary_img_global = cv2.threshold(equalized_img, 228, 255, cv2.THRESH_BINARY)
    plt.imshow(binary_img_global, cmap='gray')
    plt.title("Umbralización Global")
    plt.show() """

    # Comparativas umbralizacion adaptativa
    """ binary_img_adaptive = cv2.adaptiveThreshold(
        equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    plt.imshow(binary_img_adaptive, cmap='gray')
    plt.title("Umbralización Adaptativa")
    plt.show() """
    # Operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_img = cv2.erode(binary_img, kernel)
    dilated_img = cv2.dilate(eroded_img, kernel)
    
    return dilated_img

# Procesamiento de imágenes (ajustes en el manejo de errores)
base_dir = "C:/Users/Jose/Downloads/archive/Training"
categories = ["meningioma","notumor"]

features, labels = [], []

for category in categories:
    category_dir = os.path.join(base_dir, category)
    for filename in os.listdir(category_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(category_dir, filename)
            try:
                print("procesando imagen:"+filename)
                processed_img = preprocess_image(image_path)
                
                # Verificación adicional de imagen binaria
                if np.all(processed_img == 0) or np.all(processed_img == 255):
                    print(f"Imagen binaria uniforme: {image_path}")
                    continue
                
                glcm_features = extract_glcm_features(processed_img)
                features.append(glcm_features)
                labels.append(1 if category == "meningioma" else 0)
                
            except Exception as e:
                print(f"Omitiendo {filename}: {str(e)}")

# Convertir y guardar datos
features = np.array(features)
labels = np.array(labels)

np.save("glcm_features.npy", features)
np.save("labels.npy", labels)

print(f"\nResumen final:")
print(f"- Características extraídas: {features.shape}")
print(f"- Imágenes fallidas: {len(os.listdir(category_dir))*2 - len(features)}")