import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Configuración inicial
BASE_DIR = "D:/Google Drive/universidad/Tesis/Codigos/python/CNN_ANFIS/archive/test"
NUM_IMAGES_COMPARE = 15  # Número de imágenes para visualización comparativa
np.random.seed(42)

def metodo1_preprocess(image_path):
    """Implementación del primer método de preprocesamiento"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img)
        
        # Filtro Bilateral
        img_bilateral = cv2.bilateralFilter(img_clahe, 9, 75, 75)
        
        # Umbralización Adaptativa
        img_thresh = cv2.adaptiveThreshold(
            img_bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 5)
        
        # Morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return img_morph
    except:
        return None

def metodo2_preprocess(image_path):
    """Implementación del segundo método de preprocesamiento"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Ecualización
        equalized_img = cv2.equalizeHist(img)
        
        # Umbralización Adaptativa
        binary_adaptive = cv2.adaptiveThreshold(
            equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 4)
        
        # Morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        opened = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    except:
        return None

def extract_features(image):
    """Función unificada para extracción de características"""
    if image is None or np.std(image) == 0:
        return None
    
    try:
        glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
        variance = graycoprops(glcm, 'variance')[0, 0]
        return np.array([contrast, asm, homogeneity, energy, entropy, variance])
    except:
        return None

def compare_preprocessing_methods(image_paths):
    """Función principal de comparación"""
    # Seleccionar muestra aleatoria para visualización
    sample_paths = np.random.choice(image_paths, NUM_IMAGES_COMPARE, replace=False)
    
    # 1. Comparación visual
    plt.figure(figsize=(20, 4*NUM_IMAGES_COMPARE))
    for i, path in enumerate(sample_paths):
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        m1 = metodo1_preprocess(path)
        m2 = metodo2_preprocess(path)
        
        plt.subplot(NUM_IMAGES_COMPARE, 3, 3*i+1)
        plt.imshow(original, cmap='gray')
        plt.title(f'Original\n{os.path.basename(path)}')
        
        plt.subplot(NUM_IMAGES_COMPARE, 3, 3*i+2)
        plt.imshow(m1 if m1 is not None else np.zeros_like(original), cmap='gray')
        plt.title('Método 1')
        
        plt.subplot(NUM_IMAGES_COMPARE, 3, 3*i+3)
        plt.imshow(m2 if m2 is not None else np.zeros_like(original), cmap='gray')
        plt.title('Método 2')
    
    plt.tight_layout()
    plt.savefig('comparacion_metodos.png')
    plt.close()
    
    # 2. Comparación estadística de características
    features_m1 = []
    features_m2 = []
    errors_m1 = 0
    errors_m2 = 0
    
    for i, path in enumerate(image_paths):
        # Procesar con ambos métodos
        img_m1 = metodo1_preprocess(path)
        img_m2 = metodo2_preprocess(path)
        
        # Extraer características
        if img_m1 is not None:
            feat = extract_features(img_m1)
            if feat is not None:
                features_m1.append(feat)
            else:
                errors_m1 += 1
        else:
            errors_m1 += 1
            
        if img_m2 is not None:
            feat = extract_features(img_m2)
            if feat is not None:
                features_m2.append(feat)
            else:
                errors_m2 += 1
        else:
            errors_m2 += 1
    
    # Convertir a arrays numpy
    features_m1 = np.array(features_m1)
    features_m2 = np.array(features_m2)
    
    # 3. Análisis estadístico
    print("\nAnálisis Estadístico Comparativo")
    print("--------------------------------")
    print(f"Método 1 - Muestras válidas: {len(features_m1)}")
    print(f"Método 2 - Muestras válidas: {len(features_m2)}")
    print(f"Errores Método 1: {errors_m1}")
    print(f"Errores Método 2: {errors_m2}\n")
    
    # Comparación por característica
    feature_names = ['Contrast', 'ASM', 'Homogeneity', 'Energy', 'Entropy', 'Variance']
    p_values = []
    
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.boxplot([features_m1[:,i], features_m2[:,i]], labels=['Método 1', 'Método 2'])
        plt.title(feature_names[i])
        
        # Test estadístico
        _, p = stats.ttest_ind(features_m1[:,i], features_m2[:,i], equal_var=False)
        p_values.append(p)
    
    plt.tight_layout()
    plt.savefig('comparacion_caracteristicas.png')
    plt.close()
    
    # Mostrar resultados estadísticos
    print("Diferencias estadísticas (test t de Welch):")
    for name, p in zip(feature_names, p_values):
        print(f"{name}: {'Diferencia significativa' if p < 0.05 else 'Sin diferencia significativa'} (p={p:.4f})")
    
    # 4. Comparación de distribuciones normalizadas
    scaler = StandardScaler()
    norm_m1 = scaler.fit_transform(features_m1)
    norm_m2 = scaler.fit_transform(features_m2)
    
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.hist(norm_m1[:,i], alpha=0.5, label='Método 1', bins=20)
        plt.hist(norm_m2[:,i], alpha=0.5, label='Método 2', bins=20)
        plt.title(feature_names[i])
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('distribuciones_normalizadas.png')
    plt.close()

# Obtener todas las rutas de imágenes
meningioma_paths = glob.glob(os.path.join(BASE_DIR, "meningioma", "Tr-me_*.jpg"))
notumor_paths = glob.glob(os.path.join(BASE_DIR, "notumor", "Tr-no_*.jpg"))
all_paths = meningioma_paths + notumor_paths

# Ejecutar comparación
compare_preprocessing_methods(all_paths)
print("Comparación completada. Ver los archivos .png generados")