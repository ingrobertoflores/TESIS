import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_HEIGHT = 36

IMG_SIZE = [100, 60]

DIGITS = "0123456789"

CHARS = "ABCDEFGHIJKLEMNOPQRSTUVWXYZ" + DIGITS + "- "


class RESOURCES:
    DIR = os.path.join(BASE_DIR, 'resources')

    FONTS = os.path.join(DIR, 'fonts')

    TRAIN = os.path.join(DIR, 'train')

    TEST = os.path.join(DIR, 'test')

    VALID = os.path.join(DIR, 'valid')

    CHECKPOINT = os.path.join(DIR, 'checkpoint')


FONT_DIR = os.path.join(RESOURCES.FONTS, 'LicensePlateUsa-55gB.ttf')

PLATE_TYPES = ['N', 'CD', 'CC', 'P', 'N', 'N', 'N', 'C', 'MI', 'MB', 'P', 'P', 'P', 'P', 'P', 'P', 'P']

VALID_PLATE_REGEX = r'^(O|N|CD|CC|P|A|C|V|PR|T|RE|AB|MI|MB|F|M|D|E)[-]{0,2}([0-9]{4,6})$'

VALID_PLATE_TXTPATTERN = "%s%s%s%s %s%s%s"

VALID_PLATE_SIZE = 7

TRAIN_STR = "train"

TEST_STR = "test"

VALID_STR = "valid"
