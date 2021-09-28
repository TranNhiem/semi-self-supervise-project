

from tensorflow.keras.optimizers import schedules
import argparse
from perceiver_compact_Conv_transformer_VIT_architecture import convnet_perceiver_architecture

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.backend import dropout, learning_phase
from tensorflow.keras import optimizers


# Setting GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[0:5], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)
strategy = tf.distribute.MirroredStrategy()


# Try to keep latten array small
input_shape = (32, 32, 3)
IMG_SIZE = 32
num_class = 100
# Patches unroll for ViT and Normal transformer
# patch_size = 4
# num_patches = (IMG_SIZE//patch_size)**2
# data_dim = num_patches

num_conv_layers = 2  # for unroll patches -- Overlap
spatial2projection_dim = [128, 256]  # This equivalent to # filters
conv_position_embedding = True
latten_dim = 128  # size of latten array --> (N)
projection_dim = 256
dropout = 0.2
stochastic_depth_rate = 0.1
# Learnable array
# (NxD) #--> OUTPUT( [Q, K][Conetent information, positional])
# latten_array = latten_dim * projection_dim

num_multi_heads = 8  # --> multhi Attention Module to processing inputs
# Encoder -- Decoder are # --> Increasing block create deeper Transformer model
NUM_TRANSFORMER_BLOCK = 4
# Corresponding with Depth of self-attention
# Model depth stack multiple CrossAttention +self-trasnformer_Block
NUM_MODEL_LAYERS = 4

# 2 layer MLP Dense with number of Unit= pro_dim
FFN_layers_units = [projection_dim, projection_dim]
classification_head = [projection_dim, num_class]

print(f"Image size: {IMG_SIZE} X {IMG_SIZE} = {IMG_SIZE ** 2}")
# print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
# print(f"Patches per image: {num_patches}")
# print(
#     f"Elements per patch [patch_size*patch_size] (3 channels RGB): {(patch_size ** 2) * 3}")
print(f"Latent array shape: {latten_dim} X {projection_dim}")
# print(f"Data array shape: {num_patches} X {projection_dim}")


with strategy.scope():

    def main(args):

        BATCH_SIZE = args.batch_size
        EPOCHS = args.num_epochs

        # Prepare data training
        #train_ds, test_ds = CIFAR100(BATCH_SIZE, IMG_SIZE)

        # Create model Architecutre
        # Noted of Input pooling mode 2D not support in current desing ["1D","sequence_pooling" ]
        conv_perceiver_model = convnet_perceiver_architecture(IMG_SIZE, num_conv_layers,  conv_position_embedding, spatial2projection_dim,
                                                              latten_dim, projection_dim, num_multi_heads,
                                                              NUM_TRANSFORMER_BLOCK, NUM_MODEL_LAYERS, FFN_layers_units, dropout,
                                                              classification_head, include_top=True, pooling_mode="1D",
                                                              stochastic_depth=False, stochastic_depth_rate=stochastic_depth_rate)

        conv_perceiver_model(tf.keras.Input((input_shape)))
        conv_perceiver_model.summary()

        # Initialize the Random weight
        x = tf.random.normal((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        h = conv_perceiver_model(x, training=False)
        print("Succeed Initialize online encoder")
        print(f"Conv_Perciever encoder OUTPUT: {h.shape}")

        num_params_f = tf.reduce_sum(
            [tf.reduce_prod(var.shape) for var in conv_perceiver_model.trainable_variables])
        print('The encoders have {} trainable parameters each.'.format(num_params_f))

        '''

        # Model Hyperparameter Defined
        # 1. Define init
        init_lr = 1e-3
        weight_decay = 1e-6
        # 2. Schedule init
        step = tf.Variable(0, trainable=False)
        schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            [10000, 15000], [1e-0, 1e-1, 1e-2])
        lr_schedule = 1e-3*schedule(step)
        def weight_decay_sche(): return 1e-4 * schedule(step)

        # 3. Schedule CosineDecay Define init
        steps_lr = EPOCHS*(50000/BATCH_SIZE)  # Len images/batch_size
        lr_decay_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate=0.003, decay_steps=steps_lr)

        # optimizer = tfa.optimizers.LAMB(
        #     learning_rate=init_lr, weight_decay_rate=weight_decay_sche)

        optimizer = tfa.optimizers.SGDW(
            learning_rate=init_lr, momentum=0.9, weight_decay=weight_decay_sche)

        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=init_lr, weight_decay=weight_decay)

        # model compile
        conv_perceiver_model.compile(optimizer=optimizer,
                                loss=tf.keras.losses.CategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                                         tf.keras.metrics.TopKCategoricalAccuracy(5, name="top5_acc")])


        # MODEL TRAINING

        conv_perceiver_model.fit(train_ds, epochs=EPOCHS,
                            validation_data=test_ds,)  # callbacks=callbacks_list,

        
        '''

    if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        # for training
        parser.add_argument('--encoder', type=str, required=False, default="perceiver_4", choices=[
                            'perceiver_4', 'perceiver_8', 'perceiver_12', 'perceiver_16'], help='Encoder architecture')
        parser.add_argument('--num_epochs', type=int,
                            default=500, help='Number of epochs')
        parser.add_argument('--batch_size', type=int,
                            default=200, help='Batch size for pretraining')

        # for callbackSS
        parser.add_argument('--weights_path', type=str,
                            default='./transformer_weights/Conv_perceiver_transformer_teacher_weight_SGDW.h5', help='distill_saving_weight_path')
        parser.add_argument('--logs_path', type=str,
                            default='./transformer_logs/Conv_perceiver_transformer_teacher_SGDW', help='distill_saving_weight_path')

        args = parser.parse_args()

        main(args)
