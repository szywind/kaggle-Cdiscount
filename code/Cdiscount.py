
# https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
import os, sys, math, io, cv2
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson # this is installed with the pymongo packages, not bson package, the API is different
import struct
from models import Models

import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from collections import defaultdict
from tqdm import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop, SGD

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator

# from keras.applications.inception_v3 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input as inception_preprocess_input

from resnet_152 import resnet152_model
from resnet_101 import resnet101_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

data_dir = "../input/"

train_bson_path = os.path.join(data_dir, "train.bson")
num_train_products = 7069896

# train_bson_path = os.path.join(data_dir, "train_example.bson")
# num_train_products = 82

test_bson_path = os.path.join(data_dir, "test.bson")
num_test_products = 1768182

categories_path = os.path.join(data_dir, "category_names.csv")

categories_df = pd.read_csv(categories_path, index_col="category_id")

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

categories_df.to_csv("categories.csv")
categories_df.head()

num_fold_tta = 5

def random_crop(img, dstSize, center=False):
    import random
    srcH, srcW = img.shape[:2]
    dstH, dstW = dstSize
    if srcH <= dstH or srcW <= dstW:
        return img
    if center:
        y0 = int((srcH - dstH) / 2)
        x0 = int((srcW - dstW) / 2)
    else:
        y0 = random.randrange(0, srcH - dstH)
        x0 = random.randrange(0, srcW - dstW)
    return img[y0:y0+dstH, x0:x0+dstW]

class Cdiscount():
    def __init__(self, height=180, width=180, batch_size=64, max_epochs=6, base_model='inceptionResnetV2', num_classes=5270):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.base_model = base_model
        self.num_classes = num_classes

        self.make_category_tables()
        # Check if it works:
        print(self.cat2idx[1000012755], self.idx2cat[4])

        if not os.path.exists("train_offsets.csv"):
            self.read_bson_files()
        else:
            self.train_offsets_df = pd.DataFrame.from_csv("train_offsets.csv")

        if not os.path.exists("train_images.csv") or not os.path.exists("val_images.csv"):
            self.train_val_split()
        self.data_generator()

        # models = Models(input_shape=(self.height, self.width, 3), classes=self.num_classes)
        # if self.base_model == 'vgg16':
        #     models.vgg16()
        # elif self.base_model == 'vgg19':
        #     models.vgg19()
        # elif self.base_model == 'resnet50':
        #     models.resnet50()
        # elif self.base_model == 'inceptionV3':
        #     models.inceptionV3()
        # else:
        #     print('Uknown base model')
        #     raise SystemExit
        #
        # models.compile(optimizer=RMSprop(lr=1e-4))
        # self.model = models.get_model()

        if self.base_model == 'resnet101':
            self.model = resnet101_model(self.height, self.width, color_type=3, num_classes=self.num_classes)
        elif self.base_model == 'resnet152':
            self.model = resnet152_model(self.height, self.width, color_type=3, num_classes=self.num_classes)
        else:
            models = Models(input_shape=(self.height, self.width, 3), classes=self.num_classes)
            if self.base_model == 'inceptionV4':
                models.inceptionV3()
            elif self.base_model == 'inceptionResnetV2':
                models.inceptionResnetV2()
            sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
            models.compile(optimizer=sgd)
            self.model = models.get_model()

        self.model.summary()


    def make_category_tables(self):
        self.cat2idx = {}
        self.idx2cat = {}
        for ir in categories_df.itertuples():
            category_id = ir[0]
            category_idx = ir[4]
            self.cat2idx[category_id] = category_idx
            self.idx2cat[category_idx] = category_id
        # return cat2idx, idx2cat


    def read_bson_files(self):
        def read_bson(bson_path, num_records, with_categories):
            rows = {}
            with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
                offset = 0
                while True:
                    item_length_bytes = f.read(4)
                    if len(item_length_bytes) == 0:
                        break

                    length = struct.unpack("<i", item_length_bytes)[0]

                    f.seek(offset)
                    item_data = f.read(length)
                    assert len(item_data) == length

                    item = bson.BSON.decode(item_data)
                    product_id = item["_id"]
                    num_imgs = len(item["imgs"])

                    row = [num_imgs, offset, length]
                    if with_categories:
                        row += [item["category_id"]]
                    rows[product_id] = row

                    offset += length
                    f.seek(offset)
                    pbar.update()

            columns = ["num_imgs", "offset", "length"]
            if with_categories:
                columns += ["category_id"]

            df = pd.DataFrame.from_dict(rows, orient="index")
            df.index.name = "product_id"
            df.columns = columns
            df.sort_index(inplace=True)
            return df

        self.train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)

        self.train_offsets_df.head()

        self.train_offsets_df.to_csv("train_offsets.csv")

        # How many products?
        print("# products: ", len(self.train_offsets_df))

        # How many categories?
        print("# category: ", len(self.train_offsets_df["category_id"].unique()))

        # How many images in total?
        print("# images: ", self.train_offsets_df["num_imgs"].sum())

    def train_val_split(self):
        def make_val_set(df, split_percentage=0.2, drop_percentage=0.):
            # Find the product_ids for each category.
            category_dict = defaultdict(list)
            for ir in tqdm(df.itertuples()):
                category_dict[ir[4]].append(ir[0])

            train_list = []
            val_list = []
            with tqdm(total=len(df)) as pbar:
                for category_id, product_ids in category_dict.items():
                    category_idx = self.cat2idx[category_id]

                    # Randomly remove products to make the dataset smaller.
                    keep_size = int(len(product_ids) * (1. - drop_percentage))
                    if keep_size < len(product_ids):
                        product_ids = np.random.choice(product_ids, keep_size, replace=False)

                    # Randomly choose the products that become part of the validation set.
                    val_size = int(len(product_ids) * split_percentage)
                    if val_size > 0:
                        val_ids = np.random.choice(product_ids, val_size, replace=False)
                    else:
                        val_ids = []

                    # Create a new row for each image.
                    for product_id in product_ids:
                        row = [product_id, category_idx]
                        for img_idx in range(df.loc[product_id, "num_imgs"]):
                            if product_id in val_ids:
                                val_list.append(row + [img_idx])
                            else:
                                train_list.append(row + [img_idx])
                        pbar.update()

            columns = ["product_id", "category_idx", "img_idx"]
            train_df = pd.DataFrame(train_list, columns=columns)
            val_df = pd.DataFrame(val_list, columns=columns)
            return train_df, val_df


        train_images_df, val_images_df = make_val_set(self.train_offsets_df, split_percentage=0.2,
                                                      drop_percentage=0)


        train_images_df.head()
        val_images_df.head()

        print("Number of training images:", len(train_images_df))
        print("Number of validation images:", len(val_images_df))
        print("Total images:", len(train_images_df) + len(val_images_df))


        len(train_images_df["category_idx"].unique()), len(val_images_df["category_idx"].unique())


        category_idx = 619
        num_train = np.sum(train_images_df["category_idx"] == category_idx)
        num_val = np.sum(val_images_df["category_idx"] == category_idx)
        print("# val set / # training set: ", num_val / num_train)


        train_images_df.to_csv("train_images.csv")
        val_images_df.to_csv("val_images.csv")


    def data_generator(self):
        ''' First load the lookup tables from the CSV files '''
        # categories_df = pd.read_csv("categories.csv", index_col=0)

        train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)
        train_images_df = pd.read_csv("train_images.csv", index_col=0)
        val_images_df = pd.read_csv("val_images.csv", index_col=0)

        class BSONIterator(Iterator):
            def __init__(self, bson_file, images_df, offsets_df, num_class,
                         image_data_generator, lock, target_size=(180, 180),
                         with_labels=True, batch_size=32, shuffle=True, seed=None, center=False):

                self.file = bson_file
                self.images_df = images_df
                self.offsets_df = offsets_df
                self.with_labels = with_labels
                self.samples = len(images_df)
                self.num_class = num_class
                self.image_data_generator = image_data_generator
                self.target_size = tuple(target_size)
                self.image_shape = self.target_size + (3,)
                # self.center = center

                print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

                super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
                self.lock = lock

            def _get_batches_of_transformed_samples(self, index_array):
                batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
                if self.with_labels:
                    batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

                for i, j in enumerate(index_array):
                    # Protect file and dataframe access with a lock.
                    with self.lock:
                        image_row = self.images_df.iloc[j]
                        product_id = image_row["product_id"]
                        offset_row = self.offsets_df.loc[product_id]

                        # Read this product's data from the BSON file.
                        self.file.seek(offset_row["offset"])
                        item_data = self.file.read(offset_row["length"])

                    # Grab the image from the product.
                    item = bson.BSON.decode(item_data)
                    img_idx = image_row["img_idx"]
                    bson_img = item["imgs"][img_idx]["picture"]

                    # Load the image.
                    img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

                    # Preprocess the image.
                    x = img_to_array(img)
                    # x = random_crop(x, self.target_size, center=self.center)
                    x = self.image_data_generator.random_transform(x)
                    x = x[np.newaxis, ...]
                    x = self.image_data_generator.standardize(x)
                    x = x[0]

                    # Add the image and the label to the batch (one-hot encoded).
                    batch_x[i] = x
                    if self.with_labels:
                        batch_y[i, image_row["category_idx"]] = 1

                if self.with_labels:
                    return batch_x, batch_y
                else:
                    return batch_x

            def next(self):
                with self.lock:
                    index_array = next(self.index_generator)
                # return self._get_batches_of_transformed_samples(index_array)
                return self._get_batches_of_transformed_samples(index_array[0])


        train_bson_file = open(train_bson_path, "rb")

        import threading
        lock = threading.Lock()

        self.num_train_images = len(train_images_df)
        self.num_val_images = len(val_images_df)

        print("# train data: ", self.num_train_images)
        print("# val data: ", self.num_val_images)
        # Tip: use ImageDataGenerator for data augmentation and preprocessing.
        train_datagen = ImageDataGenerator(horizontal_flip=True,
                                           preprocessing_function=preprocess_input,
                                           # shear_range=0.2,
                                           # height_shift_range=0.1,
                                           # width_shift_range=0.1,
                                           # zoom_range=[1.0, 1.2]
                                           )
        self.train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                                 self.num_classes, train_datagen, lock,
                                 batch_size=self.batch_size, shuffle=True,
                                 target_size=(self.height, self.width), center=False)

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                               self.num_classes, val_datagen, lock,
                               batch_size=self.batch_size, shuffle=True,
                               target_size=(self.height, self.width), center=True)


        # next(self.train_gen)  # warm-up
        #
        # bx, by = next(self.train_gen)
        # plt.imshow(bx[-1].astype(np.uint8))
        # cat_idx = np.argmax(by[-1])
        # cat_id = self.idx2cat[cat_idx]
        # categories_df.loc[cat_id]
        #
        #
        #
        # by = next(self.val_gen)
        # plt.imshow(bx[-1].astype(np.uint8))
        # cat_idx = np.argmax(by[-1])
        # cat_id = self.idx2cat[cat_idx]
        # categories_df.loc[cat_id]

    def train(self):
        ''' training '''
        self.model.load_weights('../weights/best_weights_{}.hdf5'.format(self.base_model))

        callbacks = [ModelCheckpoint(filepath='../weights/best_weights_{}.hdf5'.format(self.base_model),
                                     save_best_only=True,
                                     save_weights_only=True),
                     ReduceLROnPlateau(factor=0.1,
                                       patience=2,
                                       verbose=1,
                                       epsilon=1e-4),
                     EarlyStopping(min_delta=1e-4,
                                   patience=4,
                                   verbose=1)]



        init_epochs = 0
        nRepeat = 1
        for i in range(self.max_epochs):
            # gradually decrease the learning rate
            if i:
                K.set_value(self.model.optimizer.lr, 0.8 * K.get_value(self.model.optimizer.lr))
            print("lr: ", K.get_value(self.model.optimizer.lr))
            start_epoch = (i * nRepeat)
            epochs = ((i + 1) * nRepeat)

            verbose = 1

            self.model.fit_generator(generator=self.train_gen,
                                steps_per_epoch=np.ceil(self.num_train_images / float(self.batch_size)),
                                verbose=verbose,
                                validation_data=self.val_gen,
                                validation_steps=np.ceil(self.num_val_images / float(self.batch_size)),
                                initial_epoch=start_epoch + init_epochs,
                                epochs=epochs + init_epochs,
                                callbacks=callbacks)


        # from keras.models import Sequential
        # from keras.layers import Dropout, Flatten, Dense
        # from keras.layers.convolutional import Conv2D
        # from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
        #
        # self.model = Sequential()
        # self.model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(180, 180, 3)))
        # self.model.add(MaxPooling2D())
        # self.model.add(Conv2D(64, 3, padding="same", activation="relu"))
        # self.model.add(MaxPooling2D())
        # self.model.add(Conv2D(128, 3, padding="same", activation="relu"))
        # self.model.add(MaxPooling2D())
        # self.model.add(GlobalAveragePooling2D())
        # self.model.add(Dense(self.num_classes, activation="softmax"))
        #
        # self.model.compile(optimizer="adam",
        #               loss="categorical_crossentropy",
        #               metrics=["accuracy"])
        #
        # self.model.summary()
        #
        # # To train the model:
        # self.model.fit_generator(self.train_gen,
        #                     steps_per_epoch=10,  # num_train_images // batch_size,
        #                     epochs=3,
        #                     validation_data=self.val_gen,
        #                     validation_steps=10,  # num_val_images // batch_size,
        #                     workers=8,
        #                     callbacks=callbacks)

        # To evaluate on the validation set:
        #model.evaluate_generator(val_gen, steps=num_val_images // batch_size, workers=8)

    def test_ensemble(self):
        ''' test ensemble several different models '''
        model1 = resnet101_model(180, 180, color_type=3, num_classes=self.num_classes, mode=0)
        model2 = resnet101_model(160, 160, color_type=3, num_classes=self.num_classes, mode=1)
        # model3 = resnet152_model(160, 160, color_type=3, num_classes=self.num_classes)

        models = Models(input_shape=(180, 180, 3), classes=self.num_classes)
        models.inceptionResnetV2()
        model4 = models.get_model()

        model1.load_weights('../weights/best_weights_resnet101.hdf5')
        model2.load_weights('../weights/best_weights_resnet101_160crop.hdf5')
        # model3.load_weights('../weights/best_weights_resnet152_160crop.hdf5')
        model4.load_weights('../weights/best_weights_inceptionResnetV2.hdf5')

        submission_df = pd.read_csv(data_dir + "sample_submission.csv")
        submission_df.head()

        test_datagen_resnet = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_datagen_inception = ImageDataGenerator(preprocessing_function=inception_preprocess_input)

        # models = [{"model": model1, "crop": False, "datagen": test_datagen_resnet},
        #           {"model": model2, "crop": True, "datagen": test_datagen_resnet},
        #           {"model": model4, "crop": False, "datagen": test_datagen_inception}]

        models = [{"model": model1, "crop": False, "datagen": test_datagen_resnet, "weight": 0.6},
                  {"model": model2, "crop": True, "datagen": test_datagen_resnet, "weight": 0.7}]

        data = bson.decode_file_iter(open(test_bson_path, "rb"))

        with tqdm(total=num_test_products) as pbar:
            for c, d in enumerate(data):
                num_imgs = len(d["imgs"])

                avg_pred = 0
                for model in models:
                    batch_x = []

                    for i in range(num_imgs):
                        bson_img = d["imgs"][i]["picture"]

                        # Load and preprocess the image.
                        img = load_img(io.BytesIO(bson_img))#, target_size=(self.height, self.width))
                        x = img_to_array(img)

                        batch_x.append(self.preprocess(x, model))
                    batch_x = np.array(batch_x, dtype=K.floatx())

                    prediction = model["model"].predict(batch_x, batch_size=num_imgs)
                    avg_pred += model["weight"] * self.blending(prediction, 'mean')
                cat_idx = np.argmax(avg_pred)
                submission_df.iloc[c]["category_id"] = self.idx2cat[cat_idx]
                pbar.update()
        submission_df.to_csv("../submit/my_submission_ensemble{}.csv.gz".format(len(models)), compression="gzip", index=False)


    def preprocess(self, x, model):
        if model["crop"]:
            x = cv2.resize(x, (160,160))
            # x = random_crop(x, (160, 160), center=True)
        x = model["datagen"].random_transform(x)
        x = x[np.newaxis, ...]
        x = model["datagen"].standardize(x)
        return x[0]



    def test(self):
        ''' test '''
        self.model.load_weights('../weights/best_weights_{}.hdf5'.format(self.base_model))
        submission_df = pd.read_csv(data_dir + "sample_submission.csv")
        submission_df.head()

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        data = bson.decode_file_iter(open(test_bson_path, "rb"))

        with tqdm(total=num_test_products) as pbar:
            for c, d in enumerate(data):
                product_id = d["_id"]
                num_imgs = len(d["imgs"])

                batch_x = np.zeros((num_imgs, self.height, self.width, 3), dtype=K.floatx())

                for i in range(num_imgs):
                    bson_img = d["imgs"][i]["picture"]

                    # Load and preprocess the image.
                    img = load_img(io.BytesIO(bson_img), target_size=(self.height, self.width))
                    x = img_to_array(img)

                    x = test_datagen.random_transform(x)

                    x = x[np.newaxis, ...]

                    x = test_datagen.standardize(x)

                    x = x[0]
                    # Add the image to the batch.
                    batch_x[i] = x

                prediction = self.model.predict(batch_x, batch_size=num_imgs)
                avg_pred = self.blending(prediction, 'mean')
                cat_idx = np.argmax(avg_pred)

                submission_df.iloc[c]["category_id"] = self.idx2cat[cat_idx]
                pbar.update()

        submission_df.to_csv("../submit/my_submission.csv.gz", compression="gzip", index=False)



    def blending(self, prediction, mode, cutoff_lo=0.8, cutoff_hi=0.2):
        mean_pred = prediction.mean(axis=0)
        median_pred = np.median(prediction, axis=0)
        min_pred = np.min(prediction, axis=0)
        max_pred = np.max(prediction, axis=0)
        if mode == 'mean':
            return mean_pred
        elif mode == 'median':
            return median_pred
        elif mode == 'minmax_median':
            return np.where(np.all(prediction > cutoff_lo, axis=0),
                            max_pred,
                            np.where(np.all(prediction < cutoff_hi, axis=0),
                                     min_pred,
                                     median_pred))
        elif mode == 'minmax_mean':
            return np.where(np.all(prediction > cutoff_lo, axis=0),
                            max_pred,
                            np.where(np.all(prediction < cutoff_hi, axis=0),
                                     min_pred,
                                     mean_pred))

    def test_tta(self):
        ''' test '''
        self.model.load_weights('../weights/best_weights_{}.hdf5'.format(self.base_model))
        submission_df = pd.read_csv(data_dir + "sample_submission.csv")
        submission_df.head()

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        data = bson.decode_file_iter(open(test_bson_path, "rb"))

        with tqdm(total=num_test_products) as pbar:
            for c, d in enumerate(data):
                product_id = d["_id"]
                num_imgs = len(d["imgs"])

                batch_x = np.zeros((num_imgs, self.height, self.width, 3), dtype=K.floatx())

                prediction = 0
                for _ in range(num_fold_tta):
                    for i in range(num_imgs):
                        bson_img = d["imgs"][i]["picture"]

                        # Load and preprocess the image.
                        img = load_img(io.BytesIO(bson_img)) #, target_size=(self.height, self.width))
                        x = img_to_array(img)

                        # x = random_crop(x, (self.height, self.width))

                        x = test_datagen.random_transform(x)

                        x = x[np.newaxis, ...]

                        x = test_datagen.standardize(x)

                        x = x[0]
                        # Add the image to the batch.
                        batch_x[i] = x

                    # prediction += self.model.predict(batch_x, batch_size=num_imgs) / float(num_fold_tta)
                    temp = self.model.predict(batch_x, batch_size=num_imgs) / float(num_fold_tta)
                    print(temp * num_fold_tta)
                    print(temp.shape)
                    prediction += temp
                    print("[{}] prediction: ".format(j), prediction.shape)
                print(prediction)
                avg_pred = prediction.mean(axis=0)
                cat_idx = np.argmax(avg_pred)

                submission_df.iloc[c]["category_id"] = self.idx2cat[cat_idx]
                pbar.update()

        submission_df.to_csv("../submit/my_submission.csv.gz", compression="gzip", index=False)






if __name__ == "__main__":
    cdis = Cdiscount()
    # cdis.train()
    cdis.test_ensemble()