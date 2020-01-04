import settings

import shutil
import numpy
import random
import re
import os
import csv

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

__FONT__ = ImageFont.truetype(settings.FONT_DIR, settings.FONT_HEIGHT)


def crear_placas() -> list:
    placas = []
    for tipo in settings.PLATE_TYPES:
        tam = settings.VALID_PLATE_SIZE
        chars_left = tam - len(tipo)
        strn = ''.join(['9' for i in range(chars_left)])
        for number in range(int(strn)-1):
            placa = str(number+1).rjust(chars_left, '-')
            # if es_placa_valida(tipo + placa):
            placas.append(settings.VALID_PLATE_TXTPATTERN % tuple(e for e in tipo + placa))
    write_all_lines_csv(settings.CSV_PLACAS_DIR, placas)
    return placas


def es_placa_valida(placa: str) -> bool:
    if len(placa) != settings.VALID_PLATE_SIZE:
        return False
    return re.search(settings.VALID_PLATE_REGEX, placa) is not None


def generar_placa_aleatoria() -> str:
    tipo = random.choice(settings.PLATE_TYPES)
    tam = settings.VALID_PLATE_SIZE - 1
    chars_left = tam - len(tipo)
    max_num_str = ''
    for i in range(chars_left):
        max_num_str += '9'
    number = random.randint(1, int(max_num_str))
    placa = str(number).rjust(chars_left, '-')
    return settings.VALID_PLATE_TXTPATTERN % tuple(e for e in tipo + placa)


def generar_placa_valida():
    placa = generar_placa_aleatoria()
    if es_placa_valida(placa):
        return placa
    return generar_placa_aleatoria()


def generar_imagen_placa(placa: str, path_fondo: str, path_destino: str):
    with Image.open(path_fondo) as img:
        d = ImageDraw.Draw(img)
        cur_pos = settings.TEXT_POSITION
        for c in placa:
            if c is '-':
                cur_pos = (cur_pos[0] + settings.CHAR_WIDTH + settings.CHAR_PADDING, cur_pos[1])
                continue
            d.text(xy=cur_pos, text=c, font=__FONT__, fill=settings.TEXT_COLOR)
            cur_pos = (cur_pos[0] + settings.CHAR_WIDTH + settings.CHAR_PADDING, cur_pos[1])
        img.save(path_destino)


def crear_imagen_placa(placa: str) -> str:
    path_fondo = settings.FONDO_PLACA
    path_destino = os.path.join(settings.PLACAS_GENERADAS_DIR, placa+".png").replace(' ','_')
    generar_imagen_placa(placa, path_fondo, path_destino)
    return path_destino


def generar_varias_placas_aleatorias(cantidad: int) ->list:
    l = []
    for i in range(cantidad):
        l.append(generar_placa_valida())
    return l


def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(source_coords).reshape(8)
    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


# transform_points = [(0, 0), (256, 0), (new_width, height), (xshift, height)]
def transformar_imagen(img: Image, transform_points: list, new_width: int, new_height: int) -> Image:
    width, height = img.size
    coeffs = find_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)],
        transform_points)
    img = img.transform((new_width, new_height), Image.PERSPECTIVE, coeffs,
                  Image.BICUBIC)
    return img


def transformar_imagen_from_path(src_img_path: str, final_img_path: str, transform_points: list,
                                 new_width: int, new_height: int, background_src: str, mirror: bool = False) -> Image:
    img = Image.open(src_img_path)
    img = img.convert('RGBA')
    bg = Image.open(os.path.join(settings.FOTOS_CARROS_DIR,background_src))
    if mirror:
        # cambiar las x usando formula
        # x' = width - x
        new_transform_points = []
        for tp in transform_points:
            ntp = (new_width - tp[0], tp[1])
            new_transform_points.append(ntp)
        transform_points = [
            new_transform_points[1],
            new_transform_points[0],
            new_transform_points[3],
            new_transform_points[2],
        ]
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        final_img_path = final_img_path + '.mirror.png'
    img = transformar_imagen(img, transform_points, new_width, new_height)
    out = bg.convert('RGBA')
    out.paste(img, mask=img)
    out.save(final_img_path)
    return out


def get_all_lines_csv(file_path: str) -> list:
    l = []
    with open(file_path) as csvf:
        for li in csv.reader(csvf, delimiter=','):
            l.append(li)
    return l[1:]


def write_all_lines_csv(file_path: str, lines: list):
    with open(file_path, 'w') as csvf:
        csv_writer = csv.writer(csvf)
        csv_writer.writerows(lines)


def generar_placa_y_transformadas(placa: str, lista_transformaciones: list):
    src_img_path = crear_imagen_placa(placa)
    for t in lista_transformaciones:
        destino = os.path.join(settings.PLACAS_GENERADAS_TRANSFORMADAS_DIR, "%s.%s.png"%(placa, t[2])).replace(' ','_')
        transformar_imagen_from_path(src_img_path, destino, t[0], t[1][0], t[1][1], t[2])
        if settings.CREAR_IMAGENES_ESPEJO:
            transformar_imagen_from_path(src_img_path, destino, t[0], t[1][0], t[1][1], t[2], mirror=True)


def generar_placas(cantidad: int):
    # placas = generar_varias_placas_aleatorias(cantidad)
    placas = crear_placas()
    lineas = get_all_lines_csv(settings.CSV_TRANSFORMACIONES_PATH)
    transformaciones = [[[(int(l[3]), int(l[4])),
                         (int(l[5]), int(l[6])),
                         (int(l[7]), int(l[8])),
                         (int(l[9]), int(l[10]))],
                         (int(l[1]), int(l[2])), l[0]]for l in lineas]
    for placa in placas:
        # transformaciones = random.sample(transformaciones, random.randint(0, len(transformaciones)))
        generar_placa_y_transformadas(placa, transformaciones)


def get_letter_vector(char: str) -> list:
    if len(char) > 1:
        return None
    if char not in settings.CHARS:
        return None
    letter_vector = list(numpy.zeros(len(settings.CHARS), dtype=int))
    letter_vector[settings.CHARS.index(char)] = 1
    return letter_vector


def get_word_matrix(word: str):
    return [get_letter_vector(c) for c in word]


def get_letter_from_vector(vector: list) -> str:
    return settings.CHARS[list(vector).index(1)]


def get_word_from_matrix(matrix: list) -> str:
    return ''.join([get_letter_from_vector(l) for l in list(matrix)])


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
