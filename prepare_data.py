import random as ra
import os
import settings as st


imagenes = []
for r, d, f in os.walk(st.PLACAS_GENERADAS_TRANSFORMADAS_DIR):
    imagenes = [img for img in f if img.lower().endswith('.png')]

imagenes_test = []
imagenes_train = []
imagenes_crossval = []

ra.shuffle(imagenes)

for img in imagenes:
    i = ra.uniform(0., 1.)
    if i <= st.TRAIN_SAMPLE_RATIO:
        imagenes_train.append(img)
        j = ra.uniform(0.,1.)
        if j <= st.CROSSVAL_TRAIN_SAMPLE_RATIO:
            imagenes_crossval.append(img)
    else:
        imagenes_test.append(img)
        j = ra.uniform(0.,1.)
        if j <= st.CROSSVAL_TEST_SAMPLE_RATIO:
            imagenes_crossval.append(img)


print('Writing train dataset')
with open(st.TRAIN_CSV_DIR, 'w') as csv:
    csv.writelines([l + "\n" for l in imagenes_train])


print('Writing test dataset')
with open(st.TEST_CSV_DIR, 'w') as csv:
    csv.writelines([l + "\n" for l in imagenes_test])


print('Writing crossvalidation dataset')
with open(st.CROSSVAL_CSV_DIR, 'w') as csv:
    csv.writelines([l + "\n" for l in imagenes_crossval])