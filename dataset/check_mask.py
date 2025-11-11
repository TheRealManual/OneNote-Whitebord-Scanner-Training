from PIL import Image
import numpy as np

m = Image.open('dataset/masks/image_2.png')
a = np.array(m)

print('=== image_1.png Mask Analysis ===')
print(f'Shape: {a.shape}')
print(f'Mode: {m.mode}')
print(f'Unique values: {np.unique(a)}')
print(f'Number of unique values: {len(np.unique(a))}')
print(f'Stroke pixels (>127): {(a>127).sum()} ({(a>127).sum()/a.size*100:.1f}%)')
print(f'Background pixels (<=127): {(a<=127).sum()} ({(a<=127).sum()/a.size*100:.1f}%)')

if len(np.unique(a)) == 2 and set(np.unique(a)) == {0, 255}:
    print('\n✓ PERFECT! Mask is binary (0 and 255 only)')
elif m.mode != 'L':
    print(f'\n❌ PROBLEM: Mode is {m.mode}, should be L (grayscale)')
elif len(np.unique(a)) > 10:
    print(f'\n❌ PROBLEM: Has {len(np.unique(a))} gray levels, should only have 2 (0 and 255)')
else:
    print('\n⚠️ Close but not perfect')
