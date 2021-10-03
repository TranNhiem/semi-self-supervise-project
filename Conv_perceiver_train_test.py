import wandb
import os
from utils.args import parse_args
from Data_utils.datasets import SEED
from Data_utils.datasets import CIFAR100_dataset
from tensorflow.keras import optimizers
from tensorflow.python.keras.backend import dropout, learning_phase
import tensorflow_addons as tfa
from Neural_Net_Architecture.Convnet_Transformer.perceiver_compact_Conv_transformer_VIT_architecture import convnet_perceiver_architecture, Conv_Perceiver_architecture_func_v1

import argparse
from tensorflow.keras.optimizers import schedules
from Training_strategy.learning_rate_optimizer_weight_decay_schedule import WarmUpAndCosineDecay, get_optimizer
from wandb.keras import WandbCallback
import tensorflow as tf

#import tensorflow as tf
checkpoint_dir = './test_model_checkpoint/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
wandb.login()

include_top = True
# if include_top:
# tf.config.experimental_run_functions_eagerly(True)
# tf.config.run_functions_eagerly(True)

# Setting GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[0:8], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()
Auto = tf.data.experimental.AUTOTUNE

# Try to keep latten array small
input_shape = (32, 32, 3)
IMG_SIZE = 32
num_class = 100
# Patches unroll for ViT and Normal transformer
# patch_size = 4
# num_patches = (IMG_SIZE//patch_size)**2
# data_dim = num_patches

num_conv_layers = 2  # for unroll patches -- Overlap
spatial2projection_dim = [256, 512]  # This equivalent to # filters
conv_position_embedding = True
latten_dim = 250  # size of latten array --> (N)
projection_dim = 512
dropout_rate = 0.2
stochastic_depth = True
stochastic_depth_rate = 0.1
# Learnable array
# (NxD) #--> OUTPUT( [Q, K][Conetent information, positional])
# latten_array = latten_dim * projection_dim

num_multi_heads = 8  # --> multhi Attention Module to processing inputs
# Encoder -- Decoder are # --> Increasing block create deeper Transformer model
NUM_TRANSFORMER_BLOCK = 1
# Corresponding with Depth of self-attention
# Model depth stack multiple CrossAttention +self-trasnformer_Block
NUM_MODEL_LAYERS = 2

# 2 layer MLP Dense with number of Unit= pro_dim
FFN_layers_units = [projection_dim, projection_dim]
classification_head = [num_class]

print(f"Image size: {IMG_SIZE} X {IMG_SIZE} = {IMG_SIZE ** 2}")
# print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
# print(f"Patches per image: {num_patches}")
# print(
#     f"Elements per patch [patch_size*patch_size] (3 channels RGB): {(patch_size ** 2) * 3}")
print(f"Latent array shape: {latten_dim} X {projection_dim}")
# print(f"Data array shape: {num_patches} X {projection_dim}")
args = parse_args()
args = parse_args()
BATCH_SIZE_per_replica = args.train_batch_size
global_BATCH_SIZE = BATCH_SIZE_per_replica * strategy.num_replicas_in_sync
print("Global _batch_size", global_BATCH_SIZE)

with strategy.scope():

    def main(args):

        EPOCHS = args.train_epochs

        # Prepare data training
        data = CIFAR100_dataset(global_BATCH_SIZE, IMG_SIZE)
        num_images = data.num_train_images

        train_ds, test_ds = data.supervised_train_ds_test_ds()
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        test_ds = strategy.experimental_distribute_dataset(test_ds)
        # Create model Architecutre

        # Noted of Input pooling mode 2D not support in current desing ["1D","sequence_pooling" ]
        # conv_perceiver_model = convnet_perceiver_architecture(IMG_SIZE, num_conv_layers,  conv_position_embedding, spatial2projection_dim,
        #                                                       latten_dim, projection_dim, num_multi_heads,
        #                                                       NUM_TRANSFORMER_BLOCK, NUM_MODEL_LAYERS, FFN_layers_units, dropout_rate,
        #                                                       classification_head, include_top=include_top, pooling_mode="1D",
        #                                                       stochastic_depth=stochastic_depth, stochastic_depth_rate=stochastic_depth_rate)

        conv_perceiver_model = Conv_Perceiver_architecture_func_v1(input_shape, num_class, IMG_SIZE, num_conv_layers,  conv_position_embedding, spatial2projection_dim,
                                                                   latten_dim, projection_dim, num_multi_heads,
                                                                   NUM_TRANSFORMER_BLOCK, NUM_MODEL_LAYERS, FFN_layers_units, dropout_rate,
                                                                   classification_head, include_top=include_top, pooling_mode="sequence_pooling",
                                                                   stochastic_depth=stochastic_depth, stochastic_depth_rate=stochastic_depth_rate)

        conv_perceiver_model(tf.keras.Input((input_shape)))
        conv_perceiver_model.summary()

        # Initialize the Random weight
        x = tf.random.normal((BATCH_SIZE_per_replica, IMG_SIZE, IMG_SIZE, 3))
        h = conv_perceiver_model(x, training=False)
        print("Succeed Initialize online encoder")
        print(f"Conv_Perciever encoder OUTPUT: {h.shape}")

        num_params_f = tf.reduce_sum(
            [tf.reduce_prod(var.shape) for var in conv_perceiver_model.trainable_variables])
        print('The encoders have {} trainable parameters each.'.format(num_params_f))

        # Configure Logs recording during training
        '''Training Configure'''
        configs = {
            "Model_Arch": "Conv_Perceiver_arch",
            "DataAugmentation_types": "None for testing",
            "Dataset": "Cifar100",
            "IMG_SIZE": IMG_SIZE,
            "Epochs": EPOCHS,
            "Batch_size": BATCH_SIZE_per_replica,
            "Learning_rate": "1e-3*Batch_size/512",
            "Optimizer": "AdamW",
            "Model_Parameter": num_params_f,
            "SEED": SEED,
            "Loss type": "Cross_entropy_loss",
            "Conv_Unroll_patches": num_conv_layers,
            "Number_Attention_Heads, ": num_multi_heads,
            "Num_Transformer_blocks": NUM_TRANSFORMER_BLOCK,
        }

        wandb.init(project="heuristic_attention_representation_learning",
                   sync_tensorboard=True, config=configs)

        # Model Hyperparameter Defined Primary
        # 1. Define init
        # init_lr = 1e-3
        # weight_decay = 1e-6
        # # # 2. Schedule init
        # # optimizer = tfa.optimizers.LAMB(
        # #     learning_rate=init_lr, weight_decay_rate=weight_decay_sche)

        # optimizer = tfa.optimizers.SGDW(
        #     learning_rate=init_lr, momentum=0.9, weight_decay=weight_decay)

        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=init_lr, weight_decay=weight_decay)

        ################################
        # Custom Define Hyperparameter
        ################################
        # 3. Schedule CosineDecay warmup
        base_lr = 0.003
        lr_rate = WarmUpAndCosineDecay(base_lr, num_images, args)
        # optimizers = get_optimizer(lr_rate)
        # AdamW = optimizers.optimizer_weight_decay(args)
        # Borrow testing
        optimizer = tfa.optimizers.SGDW(
            learning_rate=lr_rate, weight_decay=args.weight_decay)

        # # model compile
        # conv_perceiver_model.compile(optimizer=optimizer,
        #                              loss=tf.keras.losses.CategoricalCrossentropy(),
        #                              metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
        #                                       tf.keras.metrics.TopKCategoricalAccuracy(5, name="top5_acc")])

        # # MODEL TRAINING

        # conv_perceiver_model.fit(train_ds, epochs=EPOCHS,
        #                          validation_data=test_ds, callbacks=[WandbCallback()])  # callbacks=callbacks_list,

        ########################################
        # Custom training Loop
        ########################################
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, model=conv_perceiver_model)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                    reduction=tf.keras.losses.Reduction.NONE)

        def distributed_loss(lables, predictions):
            # each GPU loss per_replica batch loss
            per_example_loss = loss_object(lables, predictions)
            # total sum loss //Global batch_size
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_BATCH_SIZE)

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

        @tf.function
        def train_step(x, y):  # (bs, 32, 32, 3), (bs)

            # Forward pass
            with tf.GradientTape() as tape:
                # (bs, 512)
                y_pred_logits = conv_perceiver_model(
                    x, training=True)  # (bs, 10)
                loss = distributed_loss(y, y_pred_logits)
            # Backward pass
            grads = tape.gradient(
                loss, conv_perceiver_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, conv_perceiver_model.trainable_variables))

            train_accuracy.update_state(y, y_pred_logits)

            return loss

        def test_step(x, y):
            images = x
            labels = y

            predictions = conv_perceiver_model(images, training=False)
            t_loss = loss_object(labels, predictions)

            test_loss.update_state(t_loss)
            test_accuracy.update_state(labels, predictions)

        @ tf.function
        def distributed_train_step(ds_one, ds_two):
            per_replica_losses = strategy.run(
                train_step, args=(ds_one, ds_two))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

        @ tf.function
        def distributed_test_step(ds_one, ds_two):
            return strategy.run(test_step, args=(ds_one, ds_two))

        for epoch_id in range(EPOCHS):
            total_loss = 0.0
            num_batches = 0
            for _, (train_x, train_y) in enumerate(train_ds):

                total_loss += distributed_train_step(train_x, train_y)
                num_batches += 1
            train_loss = total_loss/num_batches

            for _, (test_x, test_y) in enumerate(test_ds):
                distributed_test_step(test_x, test_y)
            if epoch_id % 10 == 0:
                checkpoint.save(checkpoint_prefix)

            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                        "Test Accuracy: {}")
            print(template.format(epoch_id+1, train_loss,
                                  train_accuracy.result(), test_loss.result(),
                                  test_accuracy.result()))

            wandb.log({
                "epochs": epoch_id,
                "train_loss": train_loss,
                "train_acc": train_accuracy.result(),
                "test_loss": test_loss.result(),
                "test_acc": test_accuracy.result(),
                "learning_rate": lr_rate

            })
            # train_loss.reset_states()
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

    if __name__ == '__main__':

        args = parse_args()

        main(args)
