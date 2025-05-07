import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

# CSV path
csv_path = "D:\\Salman\\Semesters\\Sem6\\Mini Project\\Mini_Project_Dataset\\_annotations.csv"
df = pd.read_csv(csv_path)

# Check columns
print(df.columns)

# Map class names to numbers
class_map = {'Microplastic': 0}
df['class'] = df['class'].map(lambda x: class_map.get(x, -1))
df = df[df['class'] != -1]  # Drop unmapped classes

# Paths
img_dir = "D:\\Salman\\Semesters\\Sem6\\Mini Project\\Mini_Project_Dataset\\all_images"
output_img_dir = "D:\\Salman\\Semesters\\Sem6\\Mini Project\\Mini_Project_Dataset\\images"
output_lbl_dir = "D:\\Salman\\Semesters\\Sem6\\Mini Project\\Mini_Project_Dataset\\labels"

# Create necessary directories
os.makedirs(os.path.join(output_img_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_img_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_lbl_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_lbl_dir, "val"), exist_ok=True)

# Get image sizes
def get_img_size(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

df['path'] = df['filename'].apply(lambda x: os.path.join(img_dir, x))
df['width'], df['height'] = zip(*df['path'].map(get_img_size))

# Train/Val Split
image_files = df['filename'].unique()
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Convert row to YOLO format
def convert_yolo(row):
    x_center = ((row['xmin'] + row['xmax']) / 2) / row['width']
    y_center = ((row['ymin'] + row['ymax']) / 2) / row['height']
    w = (row['xmax'] - row['xmin']) / row['width']
    h = (row['ymax'] - row['ymin']) / row['height']
    return f"{int(row['class'])} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

# Save images and labels
def save_split(split_files, split):
    for fname in split_files:
        src_img = os.path.join(img_dir, fname)
        dst_img = os.path.join(output_img_dir, split, fname)
        shutil.copy(src_img, dst_img)

        rows = df[df['filename'] == fname]
        label_txt = ''.join([convert_yolo(row) for _, row in rows.iterrows()])
        label_fname = fname.rsplit('.', 1)[0] + '.txt'
        with open(os.path.join(output_lbl_dir, split, label_fname), 'w') as f:
            f.write(label_txt)

# Run
save_split(train_files, 'train')
save_split(val_files, 'val')
