#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from metrics import mean_iou, mean_score, bce_dice_loss

def build_model_ref(
        height, width, channels, optimizer='adam', dice=False, out_ch=1, start_ch=16, depth=5, inc_rate=2,
        activation='relu', dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=False):
    """Copy from https://www.kaggle.com/dingli/seismic-data-analysis-with-u-net"""

    def conv_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same')(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, activation=acti, padding='same')(n)
        n = BatchNormalization()(n) if bn else n
        return Concatenate()([m, n]) if res else n

    def level_block(tensor, dimension, depth, inc_rate, activation, dropout, bacthnorm, maxpool, upconv, residual):
        if depth > 0:
            n = conv_block(tensor, dimension, activation, bacthnorm, residual)
            tensor = MaxPooling2D()(n) if maxpool else Conv2D(dimension, 3, strides=2, padding='same')(n)
            tensor = level_block(
                tensor, int(inc_rate * dimension),
                depth - 1, inc_rate, activation, dropout, bacthnorm, maxpool, upconv, residual)
            if upconv:
                tensor = UpSampling2D()(tensor)
                tensor = Conv2D(dimension, 2, activation=activation, padding='same')(tensor)
            else:
                tensor = Conv2DTranspose(dimension, 3, strides=2, activation=activation, padding='same')(tensor)
            n = Concatenate()([n, tensor])
            tensor = conv_block(n, dimension, activation, bacthnorm, residual)
        else:
            tensor = conv_block(tensor, dimension, activation, bacthnorm, residual, dropout)
        return tensor

    def UNet(img_shape, out_ch, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual):
        inputs = Input(shape=img_shape)
        outputs = level_block(
            inputs, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
        outputs = Conv2D(out_ch, 1, activation='sigmoid')(outputs)
        return Model(inputs=inputs, outputs=outputs)

    img_shape = [height, width, channels]
    model = UNet(img_shape, out_ch, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    if dice:
        loss = bce_dice_loss
    else:
        loss = 'binary_crossentropy'

    model.compile(optimizer=optimizer, loss=loss, metrics=[mean_iou, mean_score])
    return model


def build_model(height, width, channels, batch_norm=False, drop_out=0.0, optimizer='adam', dice=False):
    inputs = Input((height, width, channels))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(s)
    c1 = BatchNormalization()(c1) if batch_norm else c1
    c1 = Dropout(drop_out)(c1) if drop_out != 0 else c1
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1) if batch_norm else c1
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2) if batch_norm else c2
    c2 = Dropout(drop_out)(c2) if drop_out != 0 else c2
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2) if batch_norm else c2
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3) if batch_norm else c3
    c3 = Dropout(drop_out)(c3) if drop_out != 0 else c3
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3) if batch_norm else c3
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4) if batch_norm else c4
    c4 = Dropout(drop_out)(c4) if drop_out != 0 else c4
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4) if batch_norm else c4
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = BatchNormalization()(c5) if batch_norm else c5
    c5 = Dropout(drop_out)(c5) if drop_out != 0 else c5
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5) if batch_norm else c5

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    if dice:
        loss = bce_dice_loss
    else:
        loss = 'binary_crossentropy'

    model.compile(optimizer=optimizer, loss=loss, metrics=[mean_iou, mean_score])
    return model


if __name__ == '__main__':
    model = build_model(128, 128, 1)
    model.summary()
