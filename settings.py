import os

EPOCHS = 3

TRAIN_BATCH_SIZE = 5

TEST_BATCH_SIZE = 3

IMG_WIDTH = 256

IMG_HEIGHT = 128

IMG_CHANNELS = 1

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

DIGITS = "0123456789"

CHARS = "ABCDEFGHIJKLEMNOPQRSTUVWXYZ" + DIGITS + "-"

SINGLE_OUTPUT_SIZE = len(CHARS)

PLATE_TYPES = ['O', 'N', 'CD', 'CC', 'P', 'A', 'C', 'V', 'PR', 'T', 'RE', 'AB', 'MI', 'MB', 'F', 'M', 'D', 'E']

FONT_HEIGHT = 270  # Pixel size to which the chars are resized

TEXT_COLOR = (0, 0, 0)

TEXT_POSITION = (40, 120)

CHAR_WIDTH = 122

CHAR_PADDING = 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')

CSV_PLACAS_DIR = os.path.join(RESOURCES_DIR, 'placas.csv')

CARPETA_MODELOS_ENTRENADOS = os.path.join(RESOURCES_DIR, 'modelos_entrenados')

NOMBRE_MODELO = 'sv-anpr-cnn.model'

ARCHIVO_MODELO = os.path.join(CARPETA_MODELOS_ENTRENADOS,NOMBRE_MODELO)

FONDO_PLACA = os.path.join(RESOURCES_DIR, 'fondos_placas', 'FONDO_PLACA.png')

PLACAS_GENERADAS_DIR = os.path.join(RESOURCES_DIR, 'placas_generadas')

PLACAS_GENERADAS_TRANSFORMADAS_DIR = os.path.join(RESOURCES_DIR, 'placas_generadas_transformadas')

CSV_TRANSFORMACIONES_PATH = os.path.join(RESOURCES_DIR, 'plate_transformation_matrices.csv')

CREAR_IMAGENES_ESPEJO = True

FOTOS_CARROS_DIR = os.path.join(RESOURCES_DIR, 'fotos_carros')

FONT_DIR = os.path.join(RESOURCES_DIR, 'fonts', 'spideraysfonts_license-plate-usa', 'LicensePlateUsa-55gB.ttf')

VALID_PLATE_REGEX = r'^(O|N|CD|CC|P|A|C|V|PR|T|RE|AB|MI|MB|F|M|D|E)[-]{0,2}([0-9]{4,6})$'

VALID_PLATE_TXTPATTERN = "%s%s%s%s %s%s%s"

VALID_PLATE_SIZE = 7

TEST_CSV_DIR = os.path.join(RESOURCES_DIR, 'test.csv')

TRAIN_CSV_DIR = os.path.join(RESOURCES_DIR, 'train.csv')

CROSSVAL_CSV_DIR = os.path.join(RESOURCES_DIR, 'crossval.csv')

PLATES_TO_GENERATE = 15

TRAIN_SAMPLE_RATIO = 0.8

TEST_SAMPLE_RATIO = 1 - TRAIN_SAMPLE_RATIO

CROSSVAL_TRAIN_SAMPLE_RATIO = 0.3

CROSSVAL_TEST_SAMPLE_RATIO = 0.5


print("%s: %s"%('FONT_HEIGHT', FONT_HEIGHT))
print("%s: %s"%('BASE_DIR', BASE_DIR))
print("%s: %s"%('RESOURCES_DIR', RESOURCES_DIR))
print("%s: %s"%('FONDO_PLACA', FONDO_PLACA))
print("%s: %s"%('PLACAS_GENERADAS_DIR', PLACAS_GENERADAS_DIR))
print("%s: %s"%('PLACAS_GENERADAS_TRANSFORMADAS_DIR', PLACAS_GENERADAS_TRANSFORMADAS_DIR))
print("%s: %s"%('TEXT_COLOR', TEXT_COLOR))
print("%s: %s"%('TEXT_POSITION', TEXT_POSITION))
print("%s: %s"%('FOTOS_CARROS_DIR', FOTOS_CARROS_DIR))
print("%s: %s"%('FONT_DIR', FONT_DIR))
print("%s: %s"%('VALID_PLATE_REGEX', VALID_PLATE_REGEX))
print("%s: %s"%('VALID_PLATE_TXTPATTERN', VALID_PLATE_TXTPATTERN))
print("%s: %s"%('VALID_PLATE_SIZE', VALID_PLATE_SIZE))
print("%s: %s"%('TEST_CSV_DIR', TEST_CSV_DIR))
print("%s: %s"%('TRAIN_CSV_DIR', TRAIN_CSV_DIR))
print("%s: %s"%('CROSSVAL_CSV_DIR', CROSSVAL_CSV_DIR))
