import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    # Même fonction qu'avant
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # Même fonction qu'avant
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def scan_document(image_path, debug=True):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image introuvable.")
    orig = image.copy()
    
    # 1. Redimensionner (facilite les calculs d'aire)
    ratio = image.shape[0] / 500.0
    image_resized = cv2.resize(image, (int(image.shape[1]/ratio), 500))

    # 2. Prétraitement lourd pour isoler LA FEUILLE
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # STRATÉGIE 1 : Un flou beaucoup plus fort (11,11 au lieu de 5,5) pour "effacer" le texte interne
    gray_blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # STRATÉGIE 2 : Canny plus sensible aux bords peu contrastés (30, 100 au lieu de 75, 200)
    edged = cv2.Canny(gray_blurred, 30, 100)
    
    # Astuce : On dilate un peu les bords trouvés pour fermer les trous si le trait n'est pas continu
    edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # 3. Trouver les contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    # STRATÉGIE 3 : On calcule la surface totale de l'image (redimensionnée)
    image_area = image_resized.shape[0] * image_resized.shape[1]

    for c in contours:
        # On calcule la surface du contour
        contour_area = cv2.contourArea(c)
        
        # Si le contour fait moins de 20% de l'image, c'est sûrement le logo vert ! On ignore.
        if contour_area < 0.20 * image_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            screenCnt = approx
            break

    # --- MODE DEBUG (Affichage visuel de ce que l'ordi comprend) ---
    if debug:
        debug_img = image_resized.copy()
        if screenCnt is not None:
            # Dessine le gros rectangle trouvé en rouge épais
            cv2.drawContours(debug_img, [screenCnt], -1, (0, 0, 255), 3)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1), plt.title("Canny (Les bords vus)"), plt.imshow(edged, cmap='gray')
        plt.subplot(1, 3, 2), plt.title("Contour Validé (Rouge)"), plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        
        if screenCnt is not None:
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            plt.subplot(1, 3, 3), plt.title("Résultat (Scanner)"), plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        else:
            plt.subplot(1, 3, 3), plt.title("ÉCHEC : 4 coins introuvables"), plt.axis('off')
        
        plt.show()
        return orig if screenCnt is None else warped
    # ---------------------------------------------------------------

    if screenCnt is None:
        print("Avertissement : Impossible de détecter la feuille entière.")
        return orig

    return four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

if __name__ == "__main__":
    chemin_image = "/home/Abdelali/Pictures/med/10.jpg" # METS TON CHEMIN ICI
    
    # On lance la fonction (le debug=True affichera les 3 fenêtres de diagnostic)
    image_recadree = scan_document(chemin_image, debug=True)