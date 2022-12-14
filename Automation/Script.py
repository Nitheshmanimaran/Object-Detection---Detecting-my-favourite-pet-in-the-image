import fiftyone.zoo as foz
import fiftyone as fo
import shutil
import os
import random
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

class1 = input("Enter the class name you want to check (eg. Dog): ")
export_dir = input("Enter the path where you want to export the dataset (eg. C:/Users/user/Desktop/Project_1/data/): ")

class2 = input("Enter the class name you want to add as class2 (eg. Cat, Bird, Fish, Hamster): ")
class3 = input("Enter the class name you want to add as class3 (eg. Cat, Bird, Fish, Hamster): ")
class4 = input("Enter the class name you want to add as class4 (eg. Cat, Bird, Fish, Hamster): ")
class5 = input("Enter the class name you want to add as class5 (eg. Cat, Bird, Fish, Hamster): ")

print('**********************************************'+ '\n')
print('Exporting dataset for ' + class1 + '\n')
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    "validation",
    label_types=["detections"],
    classes=[class1],
    max_samples=5000,
    shuffle=True,
    dataset_name=class1,
)

dataset.export(
    export_dir=export_dir + "train/dog/",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="detections"
)
print("Dataset exported successfully" + " for " + class1)

print('**********************************************'+ '\n')
print('Exporting dataset for ' + class2 + '\n')
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    "validation",
    label_types=["detections"],
    classes=[class2, class3, class4, class5],
    max_samples=5000,
    shuffle=True,
    dataset_name=class2,
)

dataset.export(
    export_dir=export_dir + "train/not_dog/",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="detections"
)
print("Dataset exported successfully" + " for " + class2)

print('**********************************************'+ '\n')
print('Splitting dataset for ' + class1 + '\n')
source_1 = export_dir + "train/dog/data"
destination_1 = export_dir + "/test/dog/"
lenght_1 = len(os.listdir(source_1))

for i in range(0, int(lenght_1/2)):
    file = random.choice(os.listdir(source_1))
    shutil.move(os.path.join(source_1, file),destination_1)

print('**********************************************'+ '\n')
print('Splitting dataset for ' + class2 + '\n')
source_2 = export_dir + "train/not_dog/data"
destination_2 = export_dir + "/test/not_dog/"
lenght_2 = len(os.listdir(source_2))

for i in range(0, int(lenght_2/2)):
    file = random.choice(os.listdir(source_2))
    shutil.move(os.path.join(source_2, file),destination_2)

print('**********************************************'+ '\n')
print('Dataset split successfully' + '\n')

print('\n')

allfiles_1 = os.listdir(export_dir + "train/dog/data")

for file in allfiles_1:
    src_path = os.path.join(export_dir + "train/dog/data", file)
    dst_path = os.path.join(export_dir + "train/dog/", file)
    shutil.move(src_path, dst_path)

allfiles_2 = os.listdir(export_dir + "train/not_dog/data")

for file in allfiles_2:
    src_path = os.path.join(export_dir + "train/not_dog/data", file)
    dst_path = os.path.join(export_dir + "train/not_dog/", file)
    shutil.move(src_path, dst_path)

os.remove(export_dir + "train/dog/labels.json")
os.remove(export_dir + "train/not_dog/labels.json")
os.rmdir(export_dir + "train/dog/data")
os.rmdir(export_dir + "train/not_dog/data")

print('**********************************************'+ '\n')
print('Dataset prepared successfully' + '\n')

print('\n')
print('Dataset Preprocessing' + '\n')
train_datagenerator = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
test_datagenerator = ImageDataGenerator(rescale=1./255)
train_iterator = train_datagenerator.flow_from_directory(export_dir + "train/", class_mode='binary', batch_size=64, target_size=(200, 200))
test_iterator = test_datagenerator.flow_from_directory(export_dir + "test/", class_mode='binary', batch_size=64, target_size=(200, 200))

print('**********************************************'+ '\n')
print('Dataset Preprocessed successfully' + '\n')

print('\n')
print('Model Compiling' + '\n')

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
optimizer = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print('**********************************************'+ '\n')
print('Model Compiled successfully' + '\n')

print('\n')
print('Model Training' + '\n')

history = model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=10, validation_data=test_iterator, validation_steps=len(test_iterator), verbose=1)

print('**********************************************'+ '\n')

print('Model Trained successfully' + '\n')

print('Do you want to save the model? (y/n)')
save = input()
if save == 'y':
    model.save('model.h5')
    print('Model saved successfully' + '\n')
else:
    print('Model not saved' + '\n')
    