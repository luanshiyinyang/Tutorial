"""
Author: Zhou Chen
Date: 2020/3/20
Desc: desc
"""
import tensorflow.keras as keras
import pandas as pd


train_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True,
    shear_range=0.2,
    width_shift_range=0.1
)
valid_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.
)
test_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.
)

df_train = pd.read_csv('../data/desc_train.csv', encoding='utf8')
df_train['class'] = df_train['class'].astype(str)

train_generator = train_gen.flow_from_dataframe(
    dataframe=df_train,
    directory="",
    x_col='file_name',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("class number", train_generator.classes)
print("images number", train_generator.n)
print("steps", train_generator.n // train_generator.batch_size)


for step, (x, y) in enumerate(train_generator):
    print(x.shape)
    print(y.shape)