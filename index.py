import tensorflow as tf
from keras import layers, models
import keras as kk


#train dataset 
data_train = 'Train'
data_test = 'Test'


batch_size = 32 #numero di elementi che il modello prende per l'addestramento
img_size = (28,28) #grandezza della immagine

train_dataset = kk.preprocessing.image_dataset_from_directory(
    data_train,
    labels="inferred", # vengono messi in base al nome della cartella 
    label_mode="int", #saranno numerate
    batch_size=batch_size,
    image_size=img_size,
    validation_split=0.5,  # 50% per la validazione
    subset="training",     # Usato per il training
    seed=123               # Per rendere la divisione riproducibile
)

val_dataset = kk.preprocessing.image_dataset_from_directory(
    data_train,
    labels="inferred",       # Le etichette vengono inferite dai nomi delle cartelle
    label_mode="int",        # Le etichette saranno numeriche (int)
    batch_size=32,           # Numero di immagini per batch
    image_size=(28, 28),     # Ridimensionamento delle immagini a 28x28 (se necessario)
    validation_split=0.5,    # 50% dei dati per la validazione
    subset="validation",     
    seed=123                 # Seme per rendere la divisione riproducibile
)

# Prepara il dataset di test
test_dataset = kk.preprocessing.image_dataset_from_directory(
    data_test,
    labels="inferred",  # etichette basate sui nomi delle cartelle
    label_mode="int",   # etichette numeriche
    batch_size=32,      # batch di 32 immagini
    image_size=(28, 28) # ridimensiona le immagini a 28x28
)


# Visualizzazione dei dataset
print(f'\n\nTraining dataset: {len(train_dataset)} batches')
print(f'Validation dataset: {len(val_dataset)} batches')

#creazione del modello 

model = kk.Sequential([
    # Primo strato convoluzionale
    kk.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    kk.layers.MaxPooling2D((2, 2)),
    
    # Secondo strato convoluzionale
    kk.layers.Conv2D(64, (3, 3), activation='relu'),
    kk.layers.MaxPooling2D((2, 2)),
    
    # Terzo strato convoluzionale
    kk.layers.Conv2D(128, (3, 3), activation='relu'),
    kk.layers.MaxPooling2D((2, 2)),
    
    # Piattaforma di appiattimento
    kk.layers.Flatten(),
    
    # Strato denso
    kk.layers.Dense(128, activation='relu'),
    
    # Strato di output (numero di classi dipende dal numero di cartelle)
    kk.layers.Dense(len(train_dataset.class_names), activation='softmax')  # softmax per la classificazione multi-classe
])


# Compilazione del modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Per etichette numeriche (int)
              metrics=['accuracy'])

# Allenamento del modello
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10  # Puoi aumentare il numero di epoche se necessario (quante volte riguardera i dati del dataset)
)

# Valutazione del modello sulla validazione
val_loss, val_acc = model.evaluate(val_dataset)
print(f"\n\nValidation accuracy: {val_acc}")



# Valutazione del modello sulla validazione
val_loss, val_acc = model.evaluate(test_dataset)
print(f"\n\nValidation accuracy: {val_acc}")

#model.save("modello.h5")

""""
    Conversione da .h5 a tensorflowjs per renderlo leggibile dal browser
    python -m tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model modello.h5 web_model


"""