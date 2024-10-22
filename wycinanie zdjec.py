import os
import glob
import numpy as np
import cv2
import csv

# Ustawienia ścieżki do folderu skryptu
script_dir = os.path.dirname(os.path.abspath(__file__))

# Lista poziomów procentowych i odpowiadających etykiet (labeli)
percentages = ['0%', '3%', '10%', '30%', '100%']
labels = {'0%': 0, '3%': 1, '10%': 2, '30%': 3, '100%': 4}

# Funkcja do wycinania ROI na podstawie jasności z ograniczeniem wyszukiwania w pobliżu znalezionych punktów
def extract_roi_center(image, roi_width, roi_height, search_window=600):
    # Konwersja do skali szarości
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wymiary obrazu
    img_height, img_width = gray_image.shape

    # Wyznaczenie środka obrazu
    center_x = img_width // 2
    center_y = img_height // 2

    # Definiowanie okna wyszukiwania wokół środka obrazu
    x_start_search = max(0, center_x - search_window // 2)
    x_end_search = min(img_width, center_x + search_window // 2)
    y_start_search = max(0, center_y - search_window // 2)
    y_end_search = min(img_height, center_y + search_window // 2)

    # Wycinamy okno wyszukiwania wokół środka obrazu
    search_area = gray_image[y_start_search:y_end_search, x_start_search:x_end_search]

    # Sumowanie wartości jasności pikseli wzdłuż osi X i Y w wyciętym obszarze
    sum_x = np.sum(search_area, axis=0)  # Sumowanie w pionie (kolumny)
    sum_y = np.sum(search_area, axis=1)  # Sumowanie w poziomie (wiersze)

    # Znajdowanie indeksu środka dla osi X i Y (punktu o największej jasności)
    local_center_x = np.argmax(sum_x)
    local_center_y = np.argmax(sum_y)

    # Przesunięcie współrzędnych do oryginalnych współrzędnych obrazu
    center_x = x_start_search + local_center_x
    center_y = y_start_search + local_center_y

    # Wyznaczanie współrzędnych lewego górnego rogu wyciętego ROI, tak by centrum było na środku
    x_start = max(0, center_x - roi_width // 2)
    y_start = max(0, center_y - roi_height // 2)

    # Upewnienie się, że ROI mieści się w obrazie (przycinanie, jeśli wychodzi poza obraz)
    x_end = min(img_width, x_start + roi_width)
    y_end = min(img_height, y_start + roi_height)

    # Dopasowanie współrzędnych startowych, jeśli wycięte ROI nie ma odpowiednich wymiarów
    x_start = max(0, x_end - roi_width)
    y_start = max(0, y_end - roi_height)

    return image[y_start:y_end, x_start:x_end], (center_x, center_y)

# Ścieżka do zapisu wszystkich wyciętych ROI
output_dir = os.path.join(script_dir, 'light')
os.makedirs(output_dir, exist_ok=True)  # Tworzenie folderu, jeśli nie istnieje

# Ścieżka do pliku CSV
csv_path = os.path.join(script_dir, 'output_labels.csv')

# Tworzymy lub otwieramy plik CSV do zapisu nazw plików i etykiet
with open(csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['filename', 'label'])  # Nagłówki kolumn

    # Przetwarzanie obrazów dla każdego poziomu oświetlenia
    for percentage in percentages:
        # Wzorzec dla plików obrazów dla danego procentu
        image_pattern = os.path.join(script_dir, 'img_old', percentage, '*waypoint*.jpg')

        # Lista plików
        image_files = sorted(glob.glob(image_pattern))
        print(f"Znaleziono {len(image_files)} plików do przetworzenia w {percentage}.")

        # Przetwarzanie każdego waypointa
        for idx, image_path in enumerate(image_files):

            waypoint_name = os.path.basename(image_path)  # Zapisz nazwę pliku

            # Wczytaj obraz
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Nie udało się załadować obrazu: {image_path}")
                continue

            # Wycinanie ROI na podstawie centrum oświetlenia z ograniczeniem wyszukiwania
            roi, center_coords = extract_roi_center(image, 600, 700)

            # Zapisz wycięte ROI do folderu 'light'
            output_path = os.path.join(output_dir, waypoint_name)
            cv2.imwrite(output_path, roi)
            print(f"Zapisano wycięty ROI do {output_path}")

            # Zapisz nazwę pliku i odpowiadający label do pliku CSV
            label = labels[percentage]
            csv_writer.writerow([waypoint_name, label])

print(f"Przetwarzanie zakończone. Plik CSV zapisany w: {csv_path}")
