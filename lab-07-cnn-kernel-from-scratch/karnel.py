import numpy as np
np.set_printoptions(linewidth=150, suppress=True, precision=4)

classes = ["bird", "cat", "dog"]

# ---------- INPUT ----------
img = np.array([[ 10,  50, 200],
                [ 20,  60, 180],
                [ 30,  70, 220]], dtype=np.float32)
print("STEP 0: INPUT IMAGE (3x3)\n", img)

# normalize 0-255 -> 0-1 (standard CNN preprocessing)
x = img / 255.0
print("\nSTEP 1: NORMALIZE (divide by 255)\n", x)

# ---------- CONVOLUTION ----------
kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
k = np.flipud(np.fliplr(kernel))
print("\nSTEP 2: KERNEL (Sobel X)\n", kernel, "\nflipped:\n", k)

padded = np.pad(x, 1)
print("\nzero-padded (5x5):\n", padded)

conv = np.zeros_like(x)
for i in range(3):
    for j in range(3):
        region = padded[i:i+3, j:j+3]
        conv[i,j] = np.sum(region * k)
        print(f"\n  pixel({i},{j}) region:\n{region}\n  sum(region*k) = {conv[i,j]:.4f}")
print("\nCONV OUTPUT (feature map):\n", conv)

# ---------- ReLU ----------
relu = np.maximum(0, conv)
print("\nSTEP 3: ReLU = max(0,x)\n", relu)

# ---------- FLATTEN ----------
flat = relu.flatten()
print("\nSTEP 4: FLATTEN (3x3 -> 9 numbers)\n", flat)

# ---------- DENSE (fully connected) ----------
# 3 classes x 9 inputs. Hand-set toy weights (NOT trained).
W = np.array([
    # pos: 00    01    02    10    11    12    20    21    22
    [ 0.0,  0.0,  0.9,  0.0,  0.0,  0.6,  0.0,  0.0,  0.5],   # bird
    [ 0.0,  0.0,  0.3,  0.0,  0.0,  0.2,  0.0,  0.0,  0.3],   # cat
    [ 0.0,  0.0, -0.4,  0.0,  0.0, -0.3,  0.0,  0.0, -0.2],   # dog
])
b = np.array([0.0, 0.2, 0.3])
print("\nSTEP 5: DENSE LAYER  logits = W @ flat + b")
print("W (3x9):\n", W)
print("b:", b)
logits = W @ flat + b
for c, l in zip(classes, logits):
    print(f"  {c:5s} logit = {l:.4f}")

# ---------- SOFTMAX ----------
e = np.exp(logits - logits.max())
probs = e / e.sum()
print("\nSTEP 6: SOFTMAX = exp(l)/sum(exp(l))")
print("  exp values:", np.exp(logits - logits.max()))
for c, p in zip(classes, probs):
    print(f"  {c:5s} probability = {p:.4f}  ({p*100:.1f}%)")

# ---------- PREDICTION ----------
print("\nSTEP 7: PREDICTION =", classes[int(np.argmax(probs))].upper())