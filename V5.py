## THE FOLLOWING IS MY COMPLETE CODE FOR THE FINAL PROJECT ##
## THIS FILE CONTAINS IMPORTING DATA, MODEL, TRAINING AND CLASSIFICATION ##

#import libraries
import tensorflow as tf
from keras import layers, models
import numpy as np
import pandas as pd

# make class names
class_names = list(range(0,22))
class_names = [str(x) for x in class_names]

# import data from train and split into training and validation
training_data = tf.keras.utils.image_dataset_from_directory(
    'uw-syde572-fall23/5_shot/5_shot/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(534, 1020),
    seed=2848,
    validation_split=0.2,
    subset="training",
    class_names = class_names
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    'uw-syde572-fall23/5_shot/5_shot/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(534, 1020),
    seed=2848,
    validation_split=0.2,
    subset="validation",
    class_names = class_names
)

# set number of classes and epochs
num_classes = 22
num_epochs = 15

#base ResNet50
base_model = tf.keras.applications.resnet50.ResNet50(
    weights = 'imagenet',
    include_top = False,
    input_shape = (534, 1020, 3)
)

#create model on base
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation ='softmax')
])

#freeze ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

#unfreeze last 2 layers
for layer in base_model.layers[-2:]:
    layer.trainable = True

#compile model
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
)

#train model
history = model.fit(
    training_data,
    epochs = num_epochs,
    validation_data=validation_data
)

#load test data
test_data = tf.keras.utils.image_dataset_from_directory(
    'uw-syde572-fall23/5_shot/5_shot/test',
    batch_size = 32,
    image_size = (534, 1020),
    label_mode = None,
    shuffle = False
)

#make predictions for test data
predictions = model.predict(test_data)

#convert predictions into class labels with argmax
predicted_labels = np.argmax(predictions, axis=1)

#print labels
print(predicted_labels)

#save outputs into CSV
test_filenames = test_data.file_paths  # This gets the complete file paths
test_filenames = [path.split('/')[-1].split('.')[0] for path in test_filenames]  # Extract filenames without extension
submission = pd.DataFrame({'ID': test_filenames, 'Category': predicted_labels})
submission.to_csv('submissionV5.csv', index=False)