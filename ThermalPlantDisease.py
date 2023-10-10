import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess the new dataset
# Replace 'new_dataset_path' with the path to your new dataset
new_dataset_path = '/Users/dheerajchetti/Downloads'
new_data_generator = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values to [0, 1]
    validation_split=0.2
)

new_train_generator = new_data_generator.flow_from_directory(
    new_dataset_path,
    target_size=(224, 224),  # Adjust to match your model's input size
    batch_size=32,
    class_mode='categorical',  # Adjust depending on your problem
    subset='training'
)

new_val_generator = new_data_generator.flow_from_directory(
    new_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load your existing model
existing_model = keras.models.load_model('existing_model.h5')

# Fine-tune the model on the new data
history = existing_model.fit(
    new_train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=new_val_generator
)

# Save the updated model
existing_model.save('updated_model.h5')

