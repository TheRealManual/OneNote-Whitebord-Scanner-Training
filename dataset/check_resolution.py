from PIL import Image
import os

files = [f for f in os.listdir('dataset/images') if f.endswith(('.jpg', '.png'))]
print('Image native resolutions:')
for f in sorted(files):
    img = Image.open(f'dataset/images/{f}')
    print(f'  {f}: {img.size[0]}×{img.size[1]} ({img.size[0]*img.size[1]:,} pixels)')
