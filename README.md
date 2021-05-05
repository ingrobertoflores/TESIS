## training
generate plate images for training

`python gen_plates.py`

generate validation images

`python gen_plates.py -s .\valid -n 200`

train

`python main.py -m train`

or train with runtime-generated images

`python main.py -m train -r`

model checkpoint will be save for each `SAVE_STEPS` steps.  
validation will be perform for each `VALIDATE_EPOCHS` epochs.

## test
generate test images

`python gen_plates.py -s .\test -n 200`

restore checkpoint for test

`python main.py -m test -c [checkpioint]`

e.g

```
python main.py -m test -c .\checkpoint\LPRnet_steps8000_loss_0.069.ckpt
...
val loss: 0.31266
plate accuracy: 192-200 0.960, char accuracy: 1105-1115 0.99103
```

### test single image

to test single image and show result

`python main.py -m test -c [checkpoint] --img [image fullpath]`

e.g
```
python main.py -m test -c .\checkpoint\LPRnet_steps5000_loss_0.215.ckpt --img .\test\AW73RHW_18771.jpg
...
restore from checkpoint: .\checkpoint\LPRnet_steps5000_loss_0.215.ckpt
AM73RHW
```