"""
Author: Zhou Chen
Date: 2020/3/23
Desc: desc
"""
from model import ResNet50
from dataset import Caltech101
import tensorflow.keras as keras
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

model = ResNet50((224, 224, 3), n_classes=101)
model.compile(optimizer=keras.optimizers.Adam(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
train_generator, valid_generator, test_generator = Caltech101()
callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10),
             keras.callbacks.ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True),
             keras.callbacks.ReduceLROnPlateau()]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=1,
    callbacks=callbacks,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // valid_generator.batch_size
)

model.save_weights('final_weights.h5')
with open('his.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print(model.evaluate(test_generator, steps=test_generator.n // test_generator.batch_size))


