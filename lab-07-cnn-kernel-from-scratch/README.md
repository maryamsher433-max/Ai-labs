# Lab 07: CNN Forward Pass from Scratch

`karnel.py` walks a 3x3 image through every CNN step in plain NumPy and prints each stage:

1. Normalise pixels (0-255 -> 0-1)
2. Convolve with a Sobel-X kernel (zero padding)
3. ReLU
4. Flatten
5. Dense layer (hand-set toy weights, not trained)
6. Softmax -> class probabilities (bird / cat / dog)
7. Prediction

```bash
pip install numpy
python karnel.py
```
