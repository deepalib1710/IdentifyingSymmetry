import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
    
print(tf.config.list_physical_devices('GPU'))

data_dir = r'shapes'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        
        try: 
            img = cv2.imread(image_path)
            with Image.open(image_path) as img:
                tip = img.format.lower()
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
                
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=32, image_size=(256, 256))

data = data.map(lambda x, y: (x / 255, y))

total_size = len(data)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.2)
test_size = int(total_size * 0.1)

train = data.take(train_size).repeat()  
val = data.skip(train_size).take(val_size).repeat() 
test = data.skip(train_size + val_size).take(test_size) 

steps_per_epoch = train_size // 20  #taking batch size of 20
validation_steps = val_size // 20   #taking batch size of 20

data = data.repeat()

data_iterator = iter(data)
try:
    batch = next(data_iterator)
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(f'Label: {batch[1][idx].numpy()}')

    plt.show()
except StopIteration:
    print(f"Reached the end of the dataset.")

#model  building

model = Sequential([
    Input(shape=(256, 256, 3)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback], steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test:
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

img = cv2.imread('samplesquare.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize / 255, 0))
print(f'yhat: {yhat}', flush=True)

if yhat > 0.5: 
    print(f'Predicted class is Symmetric')
else:
    print(f'Predicted class is Asymmetric')

model.save(os.path.join('models', 'imageclassifier.h5'))
new_model = load_model(os.path.join('models', 'imageclassifier.h5'))
new_yhat = new_model.predict(np.expand_dims(resize / 255, 0))

if new_yhat > 0.5:
    print('New model prediction: Symmetric')
else:
    print('New model prediction: Asymmetric')
