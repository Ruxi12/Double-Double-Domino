import numpy as np
import cv2 as cv
from collections import Counter
import re
import os
import glob
import matplotlib.pyplot as plt
from numpy.random import uniform
import pdb

coloane = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
           10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O'}
punctaje = [-12,1,2,3,4,5,6,0,2,5,3,4,6,2,2,0,3,5,4,1,6,2,4,5,5,0,6,3,4,2,0,1,5,1,3,4,4,4,5,0,6,3,5,4,1,3,2,0,0,1,1,2,3,6,3,5,2,1,0,6,6,5,2,1,2,5,0,3,3,5,0,6,1,4,0,6,
          3,5,1,4,2,6,2,3,1,6,5,6,2,0,4,0,1,6,4,4,1,6,6,3,0]
puncte_traseu = [
        [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
        [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
        [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
        [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
        [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
        [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
        [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
        [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
        [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
        [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
        [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
    ]


def citire_date(file_path, image_index, position_index):
    file_path += str(image_index) + "_"
    if position_index < 10:
        file_path = file_path + "0"
    file_path += str(position_index)
    #try:
    photo = cv.imread(file_path + ".jpg")
    # f = open(file_path + ".txt", "r")
    # content = f.read()
    # finally:
    #     f.close()
    return photo

def show_image(title, image):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def HSV_composition(img):
    low_yellow = (41, 120, 32)
    high_yellow = (143, 255, 255)
    print("shape of image", img.shape)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
    cv.imshow('img_initial', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('mask_yellow_hsv', mask_yellow_hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return mask_yellow_hsv


def extrage_careu(image, extra):
    lower_values = (40, 120, 20)
    upper_values = (144, 255, 255)
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, lower_values, upper_values)
    result_image = cv.bitwise_and(img_hsv, img_hsv, mask=mask)
    # show_image("result_image_hsv", result_image)

    gray_result = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)
    #show_image("Gray_result", gray_result)
    image_g_blur = cv.GaussianBlur(gray_result, (5, 5), 0)
    image_m_blur = cv.medianBlur(gray_result, 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    _, thresh = cv.threshold(image_sharpened, 32, 255, cv.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=9)
    thresh = cv.dilate(thresh, kernel)
    edges = cv.Canny(thresh, 120, 400)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 1350
    height = 1350

    top_left = (max(top_left[0] - extra, 0), max(top_left[1] - extra, 0))
    bottom_right = (min(bottom_right[0] + extra, image.shape[1]), min(bottom_right[1] + extra, image.shape[0]))
    top_right = (min(top_right[0] + extra, image.shape[1]), max(top_right[1] - extra, 0))
    bottom_left = (max(bottom_left[0] - extra, 0), min(bottom_left[1] + extra, image.shape[0]))

    # image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    image_copy = cv.cvtColor(image.copy(), cv.COLOR_HSV2BGR)


    cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)
    # show_image("image_copy", image_copy)

    # print(f"top_left = {top_left}")
    # print(f"bottom_right = {bottom_right}")
    # print(f"top_right = {top_right}")
    # print(f"bottom_left = {bottom_left}")
    puzzle_corners = np.array([[top_left],[top_right],[bottom_right],[bottom_left]], dtype=np.float32)
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    perspective_transform = cv.getPerspectiveTransform(puzzle_corners, destination_of_puzzle)
    result = cv.warpPerspective(image, perspective_transform, (width, height))
    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    return result


def trasare_linii(image):
    lines_horizontal = []
    pas = 90
    for i in range(0, 1351, 90):
        l = []
        l.append((0, i))
        l.append((1349, i))
        lines_horizontal.append(l)
    #print(lines_horizontal)

    lines_vertical = []
    for i in range(0 , 1351, 90):
        l = []
        l.append((i, 0))
        l.append((i, 1349))
        lines_vertical.append(l)


    for line in lines_vertical:
        cv.line(image, line[0], line[1], (0, 255, 0), 3)
        for line in lines_horizontal:
            cv.line(image, line[0], line[1], (0, 0, 255), 3)
    # show_image("image_with_lines", image)
    return lines_horizontal, lines_vertical


offset_template = 0
def determina_configuratie_careu(img, thresh, lines_horizontal, lines_vertical):
    matrix = np.empty((15, 15), dtype='str')
    intensity_patches = []
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + offset_template
            y_max = lines_vertical[j + 1][1][0] - offset_template
            x_min = lines_horizontal[i][0][1] + offset_template
            x_max = lines_horizontal[i + 1][1][1] - offset_template
            patch = thresh[x_min:x_max, y_min:y_max].copy()
            patch_orig = img[x_min:x_max, y_min:y_max].copy()

            # cv.imwrite('thresh_patch/patch_{}_{}.jpg'.format(i, j), patch)
            # cv.imwrite('patches/patch_{}_{}.jpg'.format(i, j), patch)
            medie_patch = np.mean(patch)
            intensity_patches.append([medie_patch, i, j, patch, patch_orig])
    return intensity_patches

def sort_by_intensity(patches, nr_dominos):
    patches.sort(key=lambda x: x[0], reverse=True)
    patches = patches[ : 2*nr_dominos]
    return patches

def custom_round(value, decimals=0):
    if isinstance(value, (int, float)):
        return round(value, decimals)
    elif isinstance(value, list):
        return [round(x, decimals) for x in value]
    elif isinstance(value, tuple):
        return tuple(round(x, decimals) for x in value)
    elif isinstance(value, np.ndarray):
        return np.round(value, decimals)
    else:
        raise TypeError("Unsupported data type for rounding")


def detect_nr_circles(patch):
    nr_circles = cv.HoughCircles(patch, cv.HOUGH_GRADIENT, 1, 10,  param1=51, param2=12, minRadius=7, maxRadius=20)
    if nr_circles is None:
        return 0
    else:
        nr_circles = custom_round(nr_circles)
        return len(nr_circles[0])


def am_gasit(tuple, lista_validare):
    for elem in lista_validare:
        if all(elem[i] == tuple[i] for i in range(len(tuple))):
            return True
    return False
def cautare_corect(tuple, lista_validare):
    for elem in lista_validare:
        if elem[0] == tuple[0]:
            if elem[1] == tuple[1]:
                return elem

def analiza_dictionar(dict, most_common_value):
    max_key = max(dict.keys())
    if abs(dict[max_key] - dict[most_common_value]) < 4 and max_key < 7:
        return max_key
    if most_common_value > 6:
        return 6
    return most_common_value

def variable_offset_patch(i, j, offset_value, image, lines_horizontal, lines_vertical):
    x_min = max(lines_horizontal[i][0][1] - offset_value, 0)
    x_max = min(lines_horizontal[i + 1][1][1] + offset_value, image.shape[0])
    y_min = max(lines_vertical[j][0][0] - offset_value, 0)
    y_max = min(lines_vertical[j + 1][1][0] + offset_value, image.shape[1])
    patch = image[x_min:x_max, y_min:y_max].copy()
    return patch

if __name__ == "__main__":
    extra_patch = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

    afisare_results3 = "results3"

    if not os.path.exists(afisare_results3):
        # If it doesn't exist, create it
        os.makedirs(afisare_results3)
        print(f"Folder '{afisare_results3}' created successfully.")
    else:
        print(f"Folder '{afisare_results3}' already exists.")

    if not os.path.exists("results2"):
        os.makedirs("results2")
    if not os.path.exists("results1"):
        os.makedirs("results1")
    # citire date
    for i in range(1, 6):
        print(f"Setul {i}")
        matrice_mutari = np.zeros((15, 15))
        lista_mutari = [(0, 0)]
        for j in range(1 , 21):
            photo = citire_date("testare/", i, j)
            tabela_joc = extrage_careu(photo, extra=2)
            img = cv.imread('imagini_auxiliare/01.jpg')
            #print("margins for tabla goala")
            tabela_blanc = extrage_careu(img, extra=2)
            #show_image("tabela_blanc", tabela_blanc)
            _, thresh_binary_tabela_joc = cv.threshold(tabela_joc, 170, 255, cv.THRESH_BINARY)
            _, thresh_binary_tabela_blanc = cv.threshold(tabela_blanc, 192, 255, cv.THRESH_BINARY)
            tabela_result = cv.absdiff(thresh_binary_tabela_joc, thresh_binary_tabela_blanc)
            lines_horizontal, lines_vertical = trasare_linii(thresh_binary_tabela_joc)


            # extrag fiecare casuta din tabela si ii calculez intensitatea
            patches = determina_configuratie_careu(thresh_binary_tabela_joc, tabela_result, lines_horizontal, lines_vertical)
            # sortez dupa intensitatea patch-ului
            patches = sort_by_intensity(patches, j)
            # for val in patches:
            #     print(val[1], val[2], detect_nr_circles(val[4]))
            #     # show_image("patch " + str(val[1]) + "_" + str(val[2]), val[3])
            #     show_image("orig_patch " + str(val[1]) + "_" + str(val[2]), val[4])

            # afisare task 1
            file_name = "results1/" + str(i) + "_" + str(j) + ".txt"
            with open(file_name, 'w') as file:
                result = []
                for value in patches:
                    index_i = value[1]
                    index_j = value[2]
                    if matrice_mutari[index_i][index_j] == 0:
                        matrice_mutari[index_i][index_j] = 1
                        index_j = coloane[index_j]
                        result.append((index_i + 1, index_j))
                sorted_result = sorted(result, key=lambda x: (x[0], x[1]))

                file.write(str(sorted_result[0][0]) + str(sorted_result[0][1]))
                file.write("\n")
                file.write(str(sorted_result[1][0]) + str(sorted_result[1][1]))
                file.write("\n")

            val1 = []    # liste pentru numarul de cercuri de pe placuta de domino pusa
            val2 = []
            for ind in range(len(patches)):
                lista_cercuri = []
                matrix_circles = []
                extra_patch = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
                for extra in extra_patch:  # margini
                    # print(f"margin = {extra}")
                    lista_gresite = []
                    tabla_modif_margine = extrage_careu(image=photo, extra=extra)
                    # show_image("tabla_modif_margine", tabla_modif_margine)
                    _, thresh = cv.threshold(tabla_modif_margine, 185, 255, cv.THRESH_BINARY)
                    # extrag fiecare casuta din tabela si ii calculez intensitatea
                    val = determina_configuratie_careu(tabela_joc, thresh, lines_horizontal, lines_vertical)

                    # verificare corectitudine patch-uri
                    lista_perechi = []

                    for off in [4, 5]:
                        patch = variable_offset_patch(i=patches[ind][1], j=patches[ind][2], offset_value=off, image=thresh,
                                                          lines_horizontal=lines_horizontal, lines_vertical=lines_vertical)

                        #circles = detect_nr_circles(patches[ind][4])f
                        circles = detect_nr_circles(patch)
                        lista_cercuri.append(circles)
                        # print(patches[ind][1], patches[ind][2], circles)
                        # show_image(str(patches[ind][1]) + "_" + str(patches[ind][2]), patch)

                counted_values = dict(Counter(lista_cercuri))
                # print(counted_values)
                most_common_value = max(counted_values, key=counted_values.get)
                # print(f"i = {patches[ind][1]}, j = {patches[ind][2]}, Most common value {most_common_value}")
                elem = sorted_result[0]
                if elem[0] == patches[ind][1] + 1 and coloane[patches[ind][2]] == elem[1]:
                    val1.append(most_common_value)
                elem = sorted_result[1]
                if elem[0] == patches[ind][1] + 1 and coloane[patches[ind][2]] == elem[1]:
                    val2.append(most_common_value)

            # most counted value pentru fiecare parte din domino
            counted_values = dict(Counter(val1))
            most_common_val1 = max(counted_values, key=counted_values.get)

            counted_values = dict(Counter(val2))
            most_common_val2 = max(counted_values, key=counted_values.get)

            print(f"Setul {i}, imaginea {j}")
            print(str(sorted_result[0][0]) + str(sorted_result[0][1]) + " " + str(most_common_val1) )
            print(str(sorted_result[1][0]) + str(sorted_result[1][1]) + " " + str(most_common_val2) )
            print()

            file_name2 = "results2/" + str(i) + "_"
            if j < 10:
                file_name2 += "0" + str(j) + ".txt"
            else:
                file_name2 += str(j) + ".txt"

            with open(file_name2, 'w') as file:
                file.write(str(sorted_result[0][0]) + str(sorted_result[0][1]) + " " + str(most_common_val1) + "\n")
                file.write(str(sorted_result[1][0]) + str(sorted_result[1][1]) + " " + str(most_common_val2) + "\n")
            lista_mutari.append((str(sorted_result[0][0]) + str(sorted_result[0][1]) + " " + str(most_common_val1) ,
                                 str(sorted_result[1][0]) + str(sorted_result[1][1]) + " " + str(most_common_val2)))
            # break

        # TASK 3 - OBTINERE SCOR

        # citesc fisierul cu mutari
        nume = "_mutari.txt"
        folder = "testare/"
        file_name = folder + str(i) + nume
        with open(file_name, 'r') as file:
            lines = file.readlines()
        # Preprocess the data
        data = []
        for line in lines:
            line = line.strip().split()  # Split each line by spaces
            image_name = line[0]  # Get the image name (e.g., 1_01.jpg)
            player_name = line[1]  # Get the player name (e.g., player1)
            data.append((image_name, player_name))  # Store as tuple

        # obtin lista cu ordinea jucatorilor

        ordine_jucatori = [0]
        for item in data:
            ordine_jucatori.append(item[1])

        # declarare_variabile pentru scor
        scor_curent1 = 0
        scor_curent2 = 0

        scor_final1 = 0
        scor_final2 = 0

        pozitie_curenta1 = 0
        pozitie_curenta2 = 0

        # citesc scorurile de la task 1 + 2
        folder_result = "results2/"
        # pentru fiecare runda in parte
        for j in range(1, 21):
            # print(f"Img {j}")
            file_name3 = folder_result + str(i) + "_"
            if j < 10:
                file_name3 += "0" + str(j) + ".txt"
            else:
                file_name3 += str(j) + ".txt"
            with open(file_name3, "r") as file:
                values  = file.readlines()
                piesa_mare = []
                for val in values:
                    val = val.split()
                    piesa_mare.append(val)
            tuplu = lista_mutari[j]
            val1_tuplu = tuplu[0]
            val2_tuplu = tuplu[1]
            if os.path.exists(file_name3) == False:
                print("Ma folosesc de valorile memorate in lista")
                domino1 = val1_tuplu.split()
                domino2 = val2_tuplu.split()
            domino1 = piesa_mare[0]
            domino2 = piesa_mare[1]
            print("domino1", domino1)
            print("domino2", domino2)


            # pentru piesa 1
            litera_coloana_piesa1 = domino1[0][-1]
            for key, value in coloane.items():
                if value == litera_coloana_piesa1:
                    numar_coloana_piesa1 = int(key)
                    break
            linie_piesa1 = int(re.search(r'\d+', domino1[0]).group()) - 1
            valoare_piesa1 = int(domino1[1])

            # pentru piesa 2
            litera_coloana_piesa2 = domino2[0][-1]
            for key, value in coloane.items():
                if value == litera_coloana_piesa2:
                    numar_coloana_piesa2 = int(key)
                    break
            linie_piesa2 = int(re.search(r'\d+', domino2[0]).group()) - 1
            valoare_piesa2 = int(domino2[1])
            # print("linie_piesa1", linie_piesa1)
            # print("litera coloana domino1", litera_coloana_piesa1)
            # print("numar_coloana", numar_coloana_piesa1)
            # print("valoare_piesa1", valoare_piesa1)
            # print()
            # print("linie_piesa2", linie_piesa2)
            # print("litera coloana domino2", litera_coloana_piesa2)
            # print("numar_coloana", numar_coloana_piesa2)
            # print("valoare_piesa2", valoare_piesa2)

            scor_curent1 = 0
            scor_curent2 = 0

            if valoare_piesa1 == punctaje[pozitie_curenta1] or valoare_piesa2 == punctaje[pozitie_curenta1]:
                scor_curent1 += 3
                pozitie_curenta1 += 3

            if valoare_piesa1 == punctaje[pozitie_curenta2] or valoare_piesa2 == punctaje[pozitie_curenta2]:
                scor_curent2 += 3
                pozitie_curenta2 += 3

            if ordine_jucatori[j] == "player1":

                if puncte_traseu[linie_piesa1][numar_coloana_piesa1] != 0:
                    scor_curent1 += puncte_traseu[linie_piesa1][numar_coloana_piesa1]
                    pozitie_curenta1 += puncte_traseu[linie_piesa1][numar_coloana_piesa1]
                    if valoare_piesa1 == valoare_piesa2:
                        scor_curent1 += puncte_traseu[linie_piesa1][numar_coloana_piesa1]
                        pozitie_curenta1 += puncte_traseu[linie_piesa1][numar_coloana_piesa1]

                if puncte_traseu[linie_piesa2][numar_coloana_piesa2] != 0:
                    scor_curent1 += puncte_traseu[linie_piesa2][numar_coloana_piesa2]
                    pozitie_curenta1 += puncte_traseu[linie_piesa2][numar_coloana_piesa2]
                    if valoare_piesa1 == valoare_piesa2:
                        scor_curent1 += puncte_traseu[linie_piesa2][numar_coloana_piesa2]
                        pozitie_curenta1 += puncte_traseu[linie_piesa2][numar_coloana_piesa2]
            else:

                if puncte_traseu[linie_piesa1][numar_coloana_piesa1] != 0:
                    scor_curent2 += puncte_traseu[linie_piesa1][numar_coloana_piesa1]
                    pozitie_curenta2 += puncte_traseu[linie_piesa1][numar_coloana_piesa1]
                    if valoare_piesa1 == valoare_piesa2:
                        scor_curent2 += puncte_traseu[linie_piesa1][numar_coloana_piesa1]
                        pozitie_curenta2 += puncte_traseu[linie_piesa1][numar_coloana_piesa1]

                if puncte_traseu[linie_piesa2][numar_coloana_piesa2] != 0:
                    scor_curent2 += puncte_traseu[linie_piesa2][numar_coloana_piesa2]
                    pozitie_curenta2 += puncte_traseu[linie_piesa2][numar_coloana_piesa2]
                    if valoare_piesa1 == valoare_piesa2:
                        scor_curent2 += puncte_traseu[linie_piesa2][numar_coloana_piesa2]
                        pozitie_curenta2 += puncte_traseu[linie_piesa2][numar_coloana_piesa2]

            scor_final1 += scor_curent1
            scor_final2 += scor_curent2
            #

            afisare_results3 = "results3/" + str(i) + "_"

            if j < 10:
                afisare_results3 += "0" + str(j) + ".txt"
            else:
                afisare_results3 += str(j) + ".txt"

            with open(afisare_results3, 'w') as file3:
                file3.write(str(domino1[0]) + " " + str(domino1[1]))
                file3.write("\n")
                file3.write(str(domino2[0]) + ' ' + str(domino2[1]))
                file3.write("\n")
                if (ordine_jucatori[j] == 'player2'):
                    file3.write(str(scor_curent2))
                    print(f"Img {j}, scor {scor_curent2}")
                else:
                    file3.write(str(scor_curent1))
                    print(f"Img {j}, scor {scor_curent1}")
