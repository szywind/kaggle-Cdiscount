from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.layers import Flatten, Dense, GlobalAveragePooling2D

from loss import focal_loss

class Models:
    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes
        self.model = Sequential()

    def vgg16(self):
        base_model = VGG16(include_top=False, weights='imagenet',
                           input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def vgg19(self):
        base_model = VGG19(include_top=False, weights='imagenet',
                           input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def resnet50(self):
        base_model = ResNet50(include_top=False, weights='imagenet',
                              input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def inceptionV3(self):
        base_model = InceptionV3(include_top=False, weights='imagenet',
                                 input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='softmax'))

    def inceptionResnetV2(self):
        base_model = InceptionResNetV2(include_top=False, weights='imagenet',
                                       input_shape=self.input_shape)
        base_model.trainable = True
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='softmax'))

    def xception(self):
        base_model = Xception(include_top=False, weights='imagenet',
                                 input_shape=self.input_shape)
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='softmax'))

    def compile(self, optimizer):
        print(self.model.summary())
        # self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=optimizer, loss=focal_loss, metrics=['accuracy', 'categorical_crossentropy'])


    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def get_model(self):
        return self.model