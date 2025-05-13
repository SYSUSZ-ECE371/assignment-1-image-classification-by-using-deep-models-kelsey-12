import os
import shutil
from sklearn.model_selection import train_test_split

raw_data_path = 'flower_dataset'
output_path = 'organized_flower_dataset'

# 创建目录结构
os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)

# 类别列表
classes = sorted([d for d in os.listdir(raw_data_path) 
                 if os.path.isdir(os.path.join(raw_data_path, d))])

# 写入classes.txt
with open(os.path.join(output_path, 'classes.txt'), 'w') as f:
    f.write('\n'.join(classes))

# 准备训练集和验证集
train_file = open(os.path.join(output_path, 'train.txt'), 'w')
val_file = open(os.path.join(output_path, 'val.txt'), 'w')

for class_idx, class_name in enumerate(classes):
    os.makedirs(os.path.join(output_path, 'train', class_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val', class_name), exist_ok=True)
    
    class_dir = os.path.join(raw_data_path, class_name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
    for img in train_images:
        shutil.copy2(os.path.join(class_dir, img), 
                   os.path.join(output_path, 'train', class_name, img))
        train_file.write(f"{class_name}/{img} {class_idx}\n")
    
    for img in val_images:
        shutil.copy2(os.path.join(class_dir, img), 
                   os.path.join(output_path, 'val', class_name, img))
        val_file.write(f"{class_name}/{img} {class_idx}\n")

train_file.close()
val_file.close()

print("数据集准备完成！")
print(f"总类别数: {len(classes)}")
for cls in classes:
    train_count = len(os.listdir(os.path.join(output_path, 'train', cls)))
    val_count = len(os.listdir(os.path.join(output_path, 'val', cls)))
    print(f"{cls}: 训练集 {train_count}张, 验证集 {val_count}张")
