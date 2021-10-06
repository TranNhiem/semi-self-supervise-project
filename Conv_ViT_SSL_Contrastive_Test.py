import wandb
import os
from utils.args import parse_args
from Data_utils.datasets import SEED
from Data_utils.imagenet_data import imagenet_dataset
from Data_utils.datasets import CIFAR100_dataset
import tensorflow_addons as tfa
# Noted in conv_transform_VIT_V1_model sequence 1D working well expecept Sequence Pooling has issue
from Neural_Net_Architecture.Convnet_Transformer.perceiver_compact_Conv_transformer_VIT_architecture import conv_VIT_V1_func
from losses.self_supervised_losses import nt_xent_asymetrize_loss_v2
import argparse
from tensorflow.keras.optimizers import schedules
from Training_strategy.learning_rate_optimizer_weight_decay_schedule import WarmUpAndCosineDecay, get_optimizer
from wandb.keras import WandbCallback
import tensorflow as tf


# import tensorflow as tf
checkpoint_dir = './test_model_checkpoint/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
wandb.login()


Auto = tf.data.experimental.AUTOTUNE

# Setting GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[0:8], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()

args = parse_args()
# Image_Size for Training and Finetune --> Check IMG_SIZE correct
# Tiny ImageNet-SIZE
if args.SSL_training == "ssl_training":
    input_shape = (64, 64, 3)
    IMG_SIZE = 64

# Cifar 100 Cifar 10 -IMG-SIZE
elif args.SSL_training == "classify_test":
    input_shape = (32, 32, 3)
    IMG_SIZE = 32
# Supervised Training
num_class = 256

# Patches unroll for ViT and Normal transformer
num_conv_layers = 2  # for unroll patches -- Overlap
spatial2projection_dim = [128, 256]  # This equivalent to # filters
position_embedding_option = True

# Classification
include_top = False
stochastic_depth = False
projection_dim = 256
dropout_rate = 0.2
stochastic_depth_rate = 0.1


num_multi_heads = 8  # --> multhi Attention Module to processing inputs
# Encoder -- Decoder are # --> Increasing block create deeper Transformer model
NUM_TRANSFORMER_BLOCK = 4


# 2 layer MLP Dense with number of Unit= pro_dim
FFN_layers_units = [projection_dim, projection_dim]
classification_unit = [projection_dim, num_class]

print(f"Image size: {IMG_SIZE} X {IMG_SIZE} = {IMG_SIZE ** 2}")
temperature = 0.1

BATCH_SIZE_per_replica = args.train_batch_size
global_BATCH_SIZE = BATCH_SIZE_per_replica * strategy.num_replicas_in_sync
print("Global _batch_size", global_BATCH_SIZE)


if args.SSL_training == "ssl_train":

    # # Prepare data training
    image_path = "/data/home/Rick/Desktop/tiny_imagenet_200/train/"
    data = imagenet_dataset(IMG_SIZE, global_BATCH_SIZE, img_path=image_path)
    num_images = data.num_images
    train_ds = data.ssl_Simclr_Augment_policy()
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    # ds_two = strategy.experimental_distribute_dataset(test_ds)

if args.SSL_training == "classify_train":
    # # Prepare data training
    data = CIFAR100_dataset(global_BATCH_SIZE, IMG_SIZE)
    num_images = data.num_train_images
    train_ds, test_ds = data.supervised_train_ds_test_ds()
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    test_ds = strategy.experimental_distribute_dataset(test_ds)

with strategy.scope():

    def main(args):

        if args.SSL_training == "ssl_train":
            EPOCHS = args.train_epochs
            include_top = False
            conv_VIT_model = conv_VIT_V1_func(input_shape, num_class, IMG_SIZE, num_conv_layers, spatial2projection_dim, position_embedding_option, projection_dim,
                                              NUM_TRANSFORMER_BLOCK, num_multi_heads,
                                              FFN_layers_units, classification_unit, dropout_rate,
                                              stochastic_depth=stochastic_depth, stochastic_depth_rate=stochastic_depth_rate,
                                              include_top=include_top, pooling_mode="sequence_pooling",
                                              )

            conv_VIT_model(tf.keras.Input((input_shape)))
            conv_VIT_model.summary()

            # Initialize the Random weight
            x = tf.random.normal(
                (BATCH_SIZE_per_replica, IMG_SIZE, IMG_SIZE, 3))
            h = conv_VIT_model(x, training=False)
            print("Succeed Initialize online encoder")
            print(f"Conv_ViT encoder OUTPUT: {h.shape}")

            num_params_f = tf.reduce_sum(
                [tf.reduce_prod(var.shape) for var in conv_VIT_model.trainable_variables])
            print('The encoders have {} trainable parameters each.'.format(num_params_f))

            # Configure Logs recording during training

            # Training Configure

            configs = {
                "Model_Arch": "Conv_ViT_arch",
                "Training mode": "Pretrain_task",
                "DataAugmentation_types": "SimCLR",
                "Dataset": "TinyImageNet",
                "IMG_SIZE": IMG_SIZE,
                "Epochs": EPOCHS,
                "Batch_size": BATCH_SIZE_per_replica,
                "Learning_rate": "Linear_scale_1e-3*Batch_size/512",
                "Optimizer": "LARS_Opt",
                "SEED": SEED,
                "Loss type": "NCE_Loss Temperature",
            }

            wandb.init(project="heuristic_attention_representation_learning",
                       sync_tensorboard=True, config=configs)

            # Model Hyperparameter Defined Primary
            # 1. Define init
            # base_lr = 1e-3
            # weight_decay = 1e-6
            # # 2. Schedule init
            # step = tf.Variable(0, trainable=False)
            # schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            #     [10000, 15000], [1e-0, 1e-1, 1e-2])
            # lr_schedule = 1e-3*schedule(step)
            # def weight_decay_sche(): return 1e-4 * schedule(step)

            # optimizer = tfa.optimizers.LAMB(
            #     learning_rate=init_lr, weight_decay_rate=weight_decay_sche)

            # optimizer = tfa.optimizers.SGDW(
            #     learning_rate=lr_rate, momentum=0.9, weight_decay=weight_decay)

            # optimizer = tfa.optimizers.AdamW(
            #     learning_rate=init_lr, weight_decay=weight_decay)

            ################################
            # Custom Define Hyperparameter
            ################################
            # 3. Schedule CosineDecay warmup
            base_lr = 0.03
            lr_rate = WarmUpAndCosineDecay(base_lr, num_images, args)
            optimizers = get_optimizer(lr_rate)
            LARSW_GC = optimizers.optimizer_weight_decay_gradient_centralization(
                args)
            # Borrow testing
            # optimizer = tfa.optimizers.AdamW(
            #     learning_rate=lr_rate, weight_decay=args.weight_decay)

            checkpoint = tf.train.Checkpoint(
                optimizer=LARSW_GC, model=conv_VIT_model)

            ##########################################
            # Custom Keras Loss
            ##########################################

            def distributed_loss(x1, x2):
                # each GPU loss per_replica batch loss
                per_example_loss = nt_xent_asymetrize_loss_v2(
                    x1, x2, temperature, BATCH_SIZE_per_replica)
                # total sum loss //Global batch_size
                # return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_BATCH_SIZE)
                return tf.reduce_sum(per_example_loss) * (1./global_BATCH_SIZE)
            test_loss = tf.keras.metrics.Mean(name='test_loss')
            train_loss = tf.keras.metrics.Mean(name="train_loss")

            @tf.function
            def train_step(ds_one, ds_two):  # (bs, 32, 32, 3), (bs)

                # Forward pass
                with tf.GradientTape() as tape:
                    # (bs, 512)
                    rep_ds1 = conv_VIT_model(ds_one, training=True)  # (bs, 10)
                    rep_ds2 = conv_VIT_model(ds_two, training=True)  # (bs, 10)

                    loss = distributed_loss(rep_ds1, rep_ds2)

                # Backward pass
                grads = tape.gradient(loss, conv_VIT_model.trainable_variables)
                LARSW_GC.apply_gradients(
                    zip(grads, conv_VIT_model.trainable_variables))

                # train_accuracy.update_state(y, y_pred_logits)
                return loss

            @ tf.function
            def distributed_train_step(ds_one, ds_two):
                per_replica_losses = strategy.run(
                    train_step, args=(ds_one, ds_two))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                       axis=None)

            for epoch_id in range(EPOCHS):

                total_loss = 0.0
                num_batches = 0
                for _, (train_ds_one, train_ds_two) in enumerate(train_ds):

                    total_loss += distributed_train_step(
                        train_ds_one, train_ds_two)

                    num_batches += 1

                train_losses = total_loss/num_batches

                # # train_loss.update_state(train_losses)
                # for _, (test_ds_one, test_ds_two) in enumerate(test_ds):
                #     distributed_test_step(test_ds_one, test_ds_two)

                if epoch_id % 500 == 0:
                    checkpoint.save(checkpoint_prefix)

                template = ("Epoch {}, Train Loss: {},  ")
                print(template.format(epoch_id+1, train_losses,))

                wandb.log({
                    "epochs": epoch_id,
                    "train_loss": train_losses,
                    "learning_rate": lr_rate

                })
                # train_loss.reset_states()
                # test_loss.reset_states()

        elif args.SSL_training == "classify_train":

            EPOCHS = args.classify_epochs
            # Future Design Remove the Top Keep Only Encoder
            include_top = False

            conv_VIT_model = conv_VIT_V1_func(input_shape, num_class, IMG_SIZE, num_conv_layers, spatial2projection_dim, position_embedding_option, projection_dim,
                                              NUM_TRANSFORMER_BLOCK, num_multi_heads,
                                              FFN_layers_units, classification_unit, dropout_rate,
                                              stochastic_depth=False, stochastic_depth_rate=stochastic_depth_rate,
                                              include_top=include_top, pooling_mode="1D",
                                              )

            conv_VIT_model(tf.keras.Input((input_shape)))
            conv_VIT_model.summary()

            # Initialize the Random weight
            x = tf.random.normal(
                (BATCH_SIZE_per_replica, IMG_SIZE, IMG_SIZE, 3))
            h = conv_VIT_model(x, training=False)
            print("Succeed Initialize online encoder")
            print(f"Conv_ViT encoder OUTPUT: {h.shape}")

            num_params_f = tf.reduce_sum(
                [tf.reduce_prod(var.shape) for var in conv_VIT_model.trainable_variables])
            print('The encoders have {} trainable parameters each.'.format(num_params_f))

            # Loading self-Supervised Pretrain_weight
            checkpoint_dir = './test_model_checkpoint/'
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            conv_VIT_model.load_weights(latest)
            conv_VIT_model.trainable = False  # Freeze entire encoder model
            print("Successful Loading Model Pretrain Weight --- Freezing Encoder")

            def classification_ffn(classification_unit, dropout_rate):
                '''
                args: Layers_number_neuron  == units_neuron
                    example units_neuron=[512, 256, 256] --> layers=len(units_neuron), units= values of element inside list
                dropout rate--> adding 1 dropout percentages layer Last ffn model

                return  FFN model in keras Sequential model
                '''
                ffn_layers = []
                for units in classification_unit[:-1]:
                    ffn_layers.append(tf.keras.layers.Dense(
                        units=units, activation=tf.nn.gelu))
                    ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))
                ffn_layers.append(tf.keras.layers.Dense(
                    units=classification_unit[-1], activation='softmax'))  # activation='softmax'
                # ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))
                ffn = tf.keras.Sequential(ffn_layers)

                return ffn

            classify_model = classification_ffn(
                classification_unit, dropout_rate)
            # Configure Logs recording during training

            # Training Configure

            configs = {
                "Model_Arch": "Conv_ViT_arch",
                "DataAugmentation_types": "SimCLR",
                "Experiment_Type": "fine-Tune classification",
                "Dataset": "Cifar100",
                "IMG_SIZE": IMG_SIZE,
                "Epochs": EPOCHS,
                "Batch_size": BATCH_SIZE_per_replica,
                "Learning_rate": "1e-3*Batch_size/512",
                "Optimizer": "AdamW",
                "SEED": SEED,
                "Loss type": "Cross_entropy_loss",
            }

            wandb.init(project="heuristic_attention_representation_learning",
                       sync_tensorboard=True, config=configs)

            # Model Hyperparameter Defined Primary
            # 1. Define init
            # base_lr = 1e-3
            # weight_decay = 1e-6
            # # 2. Schedule init
            # step = tf.Variable(0, trainable=False)
            # schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            #     [10000, 15000], [1e-0, 1e-1, 1e-2])
            # lr_schedule = 1e-3*schedule(step)
            # def weight_decay_sche(): return 1e-4 * schedule(step)

            # optimizer = tfa.optimizers.LAMB(
            #     learning_rate=init_lr, weight_decay_rate=weight_decay_sche)

            # optimizer = tfa.optimizers.SGDW(
            #     learning_rate=lr_rate, momentum=0.9, weight_decay=weight_decay)

            # optimizer = tfa.optimizers.AdamW(
            #     learning_rate=init_lr, weight_decay=weight_decay)

            ################################
            # Custom Define Hyperparameter
            ################################
            # 3. Schedule CosineDecay warmup
            base_lr = 0.003
            lr_rate = WarmUpAndCosineDecay(base_lr, num_images, args)
            optimizers = get_optimizer(lr_rate)
            LARSW_GC = optimizers.optimizer_weight_decay_gradient_centralization(
                args)
            # Borrow testing
            # optimizer = tfa.optimizers.AdamW(
            #     learning_rate=lr_rate, weight_decay=args.weight_decay)

            ##########################################
            # Custom Keras Loss
            ##########################################

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
                    repr_ = conv_VIT_model(
                        x, training=False)  # (bs, 10)
                    y_pred_logits = classify_model(repr_, training=True)
                    loss = distributed_loss(y, y_pred_logits)
                # Backward pass
                grads = tape.gradient(loss, classify_model.trainable_variables)
                LARSW_GC.apply_gradients(
                    zip(grads, classify_model.trainable_variables))

                train_accuracy.update_state(y, y_pred_logits)

                return loss

            def test_step(x, y):
                images = x
                labels = y

                repr_ = conv_VIT_model(images, training=False)
                predictions = classify_model(repr_, training=False)

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
                train_losses = total_loss/num_batches
                # train_loss.update_state(train_losses)
                for _, (test_x, test_y) in enumerate(test_ds):
                    distributed_test_step(test_x, test_y)

                template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                            "Test Accuracy: {}")
                print(template.format(epoch_id+1, train_losses,
                                      train_accuracy.result(), test_loss.result(),
                                      test_accuracy.result()))

                wandb.log({
                    "epochs": epoch_id,
                    "train_loss": train_losses,
                    "train_acc": train_accuracy.result(),
                    "test_loss": test_loss.result(),
                    "test_acc": test_accuracy.result(),
                    "learning_rate": lr_rate

                })
                # train_loss.reset_states()
                test_loss.reset_states()
                train_accuracy.reset_states()
                test_accuracy.reset_states()

        else:
            raise ValueError("Training mode not Implement Yet")

    if __name__ == '__main__':

        args = parse_args()

        main(args)
