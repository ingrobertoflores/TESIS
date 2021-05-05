import settings

import shutil
import numpy as np
import random
import os
import csv
from data_aug import data_augmentation

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import LPRnet as model

import gen_plates as gen

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


def generar_imagen_placa(placa: str, path_fondo: str, path_destino: str):
    with Image.open(path_fondo) as img:
        d = ImageDraw.Draw(img)
        cur_pos = settings.TEXT_POSITION
        for c in placa:
            if c == '-':
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

    write_all_lines_csv(settings.CSV_PLACAS_DIR, l)
    return l


def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


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
    with open(file_path, 'w', newline='', encoding='utf-8') as csvf:
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
    placas = generar_varias_placas_aleatorias(cantidad)
    # placas = crear_placas()
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
    letter_vector = list(np.zeros(len(settings.CHARS), dtype=int))
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



def encode_label(label, char_dict):
    encode = [char_dict[c] for c in label]
    return encode

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

class DataIterator:
    def __init__(self, img_dir, runtime_generate=False):
        self.img_dir = img_dir
        self.batch_size = model.BATCH_SIZE
        self.channel_num = model.CH_NUM
        self.img_w, self.img_h = model.IMG_SIZE

        if runtime_generate:
            self.generator = gen.ImageGenerator(settings.RESOURCES.FONTS, model.CHARS)
        else:
            self.init()

    def init(self):
        self.filenames = []
        self.labels = []
        fs = os.listdir(self.img_dir)
        for filename in fs:
            self.filenames.append(filename)
            label = filename.split('_')[0] # format: [label]_[random number].jpg
            label = encode_label(label, model.CHARS_DICT)
            self.labels.append(label)
        self.sample_num = len(self.labels)
        self.labels = np.array(self.labels)
        self.random_index = list(range(self.sample_num))
        random.shuffle(self.random_index)
        self.cur_index = 0

    def next_sample_ind(self):
        ret = self.random_index[self.cur_index]
        self.cur_index += 1
        if self.cur_index >= self.sample_num:
            self.cur_index = 0
            random.shuffle(self.random_index)
        return ret

    def next_batch(self):

        batch_size = self.batch_size
        images = np.zeros([batch_size, self.img_h, self.img_w, self.channel_num])
        labels = []

        for i in range(batch_size):
            sample_ind = self.next_sample_ind()
            fname = self.filenames[sample_ind]
            img = cv2.imread(os.path.join(self.img_dir, fname))
            #img = data_augmentation(img)
            img = cv2.resize(img, (self.img_w, self.img_h))
            images[i] = img

            labels.append(self.labels[sample_ind])

        sparse_labels = sparse_tuple_from(labels)

        return images, sparse_labels, labels

    def next_test_batch(self):

        start = 0
        end = self.batch_size
        is_last_batch = False

        while not is_last_batch:
            if end >= self.sample_num:
                end = self.sample_num
                is_last_batch = True

            #print("s: {} e: {}".format(start, end))

            cur_batch_size = end-start
            images = np.zeros([cur_batch_size, self.img_h, self.img_w, self.channel_num])

            for j, i in enumerate(range(start, end)):
                fname = self.filenames[i]
                img = cv2.imread(os.path.join(self.img_dir, fname))
                img = cv2.resize(img, (self.img_w, self.img_h))
                images[j, ...] = img

            labels = self.labels[start:end, ...]
            sparse_labels = sparse_tuple_from(labels)

            start = end
            end += self.batch_size

            yield images, sparse_labels, labels

    def next_gen_batch(self):

        batch_size = self.batch_size
        imgs, labels = self.generator.generate_images(batch_size)
        labels = [encode_label(label, model.CHARS_DICT) for label in labels]

        images = np.zeros([batch_size, self.img_h, self.img_w, self.channel_num])
        for i, img in enumerate(imgs):
            img = data_augmentation(img)
            img = cv2.resize(img, (self.img_w, self.img_h))
            images[i, ...] = img

        sparse_labels = sparse_tuple_from(labels)

        return images, sparse_labels, labels