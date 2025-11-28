"""
Jednoduchá modulárna CNN architektúra pre American Sign Language klasifikáciu
Používa dataset: https://www.kaggle.com/datasets/kapillondhe/american-sign-language
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import List, Optional, Tuple


class SimpleCNN:
    """
    Jednoduchá modulárna CNN, ktorá sa dá ľahko rozširovať.
    
    Parametre:
    -----------
    input_shape : Tuple[int, int, int]
        Tvar vstupného obrázka (height, width, channels). Napr. (224, 224, 3)
    
    num_classes : int
        Počet výstupných tried (28 pre ASL)
    
    num_conv_layers : int
        Počet konvolučných vrstiev (default 3)
    
    filters_per_layer : List[int] alebo int
        Počet filtrov pre každú vrstvu. Ak int, použije sa rovnaký počet pre všetky.
        Ak List, musí mať rovnakú dĺžku ako num_conv_layers (default [32, 64, 128])
    
    dense_units : List[int]
        Počet neurónov v dense vrstvách (default [256, 128])
    
    dropout_rate : float
        Dropout rate pre dense vrstvy (default 0.5)
    
    use_batch_norm : bool
        Použiť BatchNormalization (default True)
    
    use_pooling : bool
        Použiť MaxPooling po každej konvolučnej vrstve (default True)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 28,
        num_conv_layers: int = 3,
        filters_per_layer: Optional[List[int]] = None,
        dense_units: Optional[List[int]] = None,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        use_pooling: bool = True
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_pooling = use_pooling
        
        # Default filtre - ak nie je zadaný zoznam, vytvoríme ho
        if filters_per_layer is None:
            # Postupné zvyšovanie filtrov: 32, 64, 128, ...
            self.filters = [32 * (2 ** i) for i in range(num_conv_layers)]
        elif isinstance(filters_per_layer, int):
            # Ak je zadané jedno číslo, použijeme ho pre všetky vrstvy
            self.filters = [filters_per_layer] * num_conv_layers
        else:
            # Ak je zadaný zoznam, použijeme ho
            if len(filters_per_layer) != num_conv_layers:
                raise ValueError(f"Počet filtrov ({len(filters_per_layer)}) sa musí rovnať počtu vrstiev ({num_conv_layers})")
            self.filters = filters_per_layer
        
        # Default dense units
        if dense_units is None:
            self.dense_units = [256, 128]
        else:
            self.dense_units = dense_units
    
    def build(self) -> keras.Model:
        """
        Vytvorí a vráti model.
        """
        # Vstupná vrstva
        inputs = layers.Input(shape=self.input_shape, name='input')
        x = inputs
        
        # Konvolučné vrstvy
        for i in range(self.num_conv_layers):
            # Conv2D vrstva
            x = layers.Conv2D(
                filters=self.filters[i],
                kernel_size=3,
                padding='same',
                activation='relu',
                name=f'conv_{i+1}'
            )(x)
            
            # Batch Normalization (ak je zapnuté)
            if self.use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            # MaxPooling (ak je zapnuté)
            if self.use_pooling:
                x = layers.MaxPooling2D(pool_size=2, name=f'pool_{i+1}')(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        # Dense vrstvy
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            
            # Dropout
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Výstupná vrstva
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Vytvorenie modelu
        model = models.Model(inputs=inputs, outputs=outputs, name='SimpleCNN')
        
        return model
    
    def compile(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: Optional[List[str]] = None
    ) -> keras.Model:
        """
        Vytvorí, skompiluje a vráti model.
        """
        model = self.build()
        
        # Nastavenie optimalizátora
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = keras.optimizers.get(optimizer)
            if hasattr(opt, 'learning_rate'):
                opt.learning_rate = learning_rate
        
        # Metriky
        if metrics is None:
            metrics = ['accuracy']
        
        # Kompilácia
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        return model


# ============================================================================
# PRÍKLAD POUŽITIA S ASL DATASETOM
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Konfigurácia (rovnaká ako v notebooku)
    DATA_DIR = '/kaggle/input/american-sign-language/ASL_Dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Train')
    TEST_DIR = os.path.join(DATA_DIR, 'Test')
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 28
    
    print("=== Príklad 1: Základný model ===")
    model1 = SimpleCNN(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        num_conv_layers=3
    )
    model1 = model1.compile()
    model1.summary()
    
    print("\n=== Príklad 2: Viac vrstiev ===")
    model2 = SimpleCNN(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        num_conv_layers=5,
        filters_per_layer=[32, 64, 128, 256, 512],
        dense_units=[512, 256, 128]
    )
    model2 = model2.compile(learning_rate=0.0001)
    model2.summary()
    
    print("\n=== Príklad 3: Jednoduchší model ===")
    model3 = SimpleCNN(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        num_conv_layers=2,
        filters_per_layer=64,
        dense_units=[128],
        dropout_rate=0.3
    )
    model3 = model3.compile()
    model3.summary()
    
    print("\n=== Príklad 4: Načítanie dát a tréning ===")
    # Načítanie dát (rovnako ako v notebooku)
    # train_data = tf.keras.utils.image_dataset_from_directory(
    #     TRAIN_DIR,
    #     image_size=IMG_SIZE,
    #     batch_size=BATCH_SIZE,
    #     color_mode='rgb',
    #     label_mode='categorical',
    #     shuffle=True
    # )
    # 
    # test_data = tf.keras.utils.image_dataset_from_directory(
    #     TEST_DIR,
    #     image_size=IMG_SIZE,
    #     batch_size=BATCH_SIZE,
    #     color_mode='rgb',
    #     label_mode='categorical',
    #     shuffle=False
    # )
    # 
    # # Normalizácia (ak je potrebná)
    # def normalize_image(image, label):
    #     image = tf.cast(image, tf.float32) / 255.0
    #     return image, label
    # 
    # train_ds = train_data.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    # test_ds = test_data.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    # 
    # # Vytvorenie a tréning modelu
    # model = SimpleCNN(
    #     input_shape=(*IMG_SIZE, 3),
    #     num_classes=NUM_CLASSES,
    #     num_conv_layers=3
    # )
    # model = model.compile()
    # 
    # # Tréning
    # history = model.fit(
    #     train_ds,
    #     epochs=10,
    #     validation_data=test_ds
    # )
    
    print("Hotovo! Model je pripravený na použitie.")
