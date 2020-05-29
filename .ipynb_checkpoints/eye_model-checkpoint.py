import tensorflow as tf
import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#LeNet-5 architecture
model = tf.keras.models.Sequential( [
    tf.keras.layers.Conv2D(6,(5,5),activation='relu',input_shape=(32,32,3)),
    #tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16,(5,5),activation='relu'),
   # tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120,activation='relu'),

    tf.keras.layers.Dense(84,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid'), 

] )

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])

model.summary()

train_datagen=ImageDataGenerator(rescale=1.0/255.0)
validation_datagen=ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    'eye_model_dataset',
    target_size=(32,32),
    batch_size=4676 ,
    class_mode='binary'
    )
validation_generator = validation_datagen.flow_from_directory(
    'eye_model_validation_dataset',
    target_size=(32,32),
    batch_size=170,
    class_mode='binary'
    )
    

model.fit_generator(train_generator,
          validation_data=validation_generator,
          steps_per_epoch=1,
          epochs=300)

model_json=model.to_json()
with open("eye_model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("eye_model.h5")
    
