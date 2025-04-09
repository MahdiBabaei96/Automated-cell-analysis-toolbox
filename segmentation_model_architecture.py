from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, MaxPooling2D, UpSampling2D, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def vgg16_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output
    s2 = vgg16.get_layer("block2_conv2").output
    s3 = vgg16.get_layer("block3_conv3").output
    s4 = vgg16.get_layer("block4_conv3").output

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    # model.summary()
    return model

def unet(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoding path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoding path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs, outputs)
    return model

def segnet(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # Decoder
    x = UpSampling2D()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

def pspnet(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)
    
    # Use a pre-trained ResNet50 model as the backbone
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    
    def pyramid_pooling_block(input_tensor, bin_sizes):
        concat_list = [input_tensor]
        w = input_tensor.shape[1]
        h = input_tensor.shape[2]

        for bin_size in bin_sizes:
            x = AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
            x = Conv2D(512, (1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = UpSampling2D(size=(w // x.shape[1] , h // x.shape[2]), interpolation='bilinear')(x)
            concat_list.append(x)
        
        return Concatenate()(concat_list)
    
    # Pyramid Pooling Module
    x = pyramid_pooling_block(resnet.output, [1, 2, 3, 6])
    
    # Final layers
    x = Conv2D(512, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D((32, 32), interpolation="bilinear")(x)
    x = Conv2D(num_classes, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)  # Changed to sigmoid for binary classification

    model = Model(inputs, x)
    return model

def DeeplabV3Plus(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)
    
    # Use a pre-trained ResNet50 model as the backbone
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # Define the atrous spatial pyramid pooling
    def aspp(input_tensor):
        shape = input_tensor.shape
        y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(input_tensor)
        y_pool = Conv2D(256, (1, 1), padding="same", use_bias=False)(y_pool)
        y_pool = BatchNormalization()(y_pool)
        y_pool = Activation("relu")(y_pool)
        y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

        y_1 = Conv2D(256, (1, 1), dilation_rate=1, padding="same", use_bias=False)(input_tensor)
        y_1 = BatchNormalization()(y_1)
        y_1 = Activation("relu")(y_1)

        y_6 = Conv2D(256, (3, 3), dilation_rate=6, padding="same", use_bias=False)(input_tensor)
        y_6 = BatchNormalization()(y_6)
        y_6 = Activation("relu")(y_6)

        y_12 = Conv2D(256, (3, 3), dilation_rate=12, padding="same", use_bias=False)(input_tensor)
        y_12 = BatchNormalization()(y_12)
        y_12 = Activation("relu")(y_12)

        y_18 = Conv2D(256, (3, 3), dilation_rate=18, padding="same", use_bias=False)(input_tensor)
        y_18 = BatchNormalization()(y_18)
        y_18 = Activation("relu")(y_18)

        y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])
        y = Conv2D(256, (1, 1), padding="same", use_bias=False)(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        return y

    # ASPP
    b4 = aspp(resnet.output)

    # Decoder
    x = UpSampling2D((4, 4), interpolation="bilinear")(b4)
    x = Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D((8, 8), interpolation="bilinear")(x)
    x = Conv2D(num_classes, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)  # Changed to sigmoid for binary classification

    model = Model(inputs, x)
    return model

def custom_conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def custom_decoder_block(input, skip_features, num_filters):
    x = UpSampling2D((2, 2))(input)
    x = Conv2D(num_filters, 2, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x = custom_conv_block(x, num_filters)
    return x

def custom_vgg16_unet(input_shape):
    inputs = Input(input_shape)

    # Pre-trained VGG16 encoder
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    # Encoder layers for skip connections
    s1 = vgg16.get_layer("block1_conv2").output  # 64 filters
    s2 = vgg16.get_layer("block2_conv2").output  # 128 filters
    s3 = vgg16.get_layer("block3_conv3").output  # 256 filters
    s4 = vgg16.get_layer("block4_conv3").output  # 512 filters
    b1 = vgg16.get_layer("block5_conv3").output  # Bridge (512 filters)

    # Decoder with custom upsampling
    d1 = custom_decoder_block(b1, s4, 512)
    d2 = custom_decoder_block(d1, s3, 256)
    d3 = custom_decoder_block(d2, s2, 128)
    d4 = custom_decoder_block(d3, s1, 64)

    # Final output layer for segmentation
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    # Construct and return the model
    model = Model(inputs, outputs, name="Custom_VGG16_U-Net")
    return model