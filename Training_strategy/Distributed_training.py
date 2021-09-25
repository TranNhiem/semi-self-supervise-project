'''
Three Important things in Distributed Training 
1.. Data_Processing for Mutli-GPUS
#Reference Discuss more in Slide material prepare tensorflow_hardware_software_utilization

--> Data_processing
    + Single Machine Data throughput
    1.- Python data Generator (the less utilize hardware -- is not efficient)
    2. - tf.data.Datset 
    3.- tf.data.Dataset + tf.function (@tf.function training loop)
    4. - tf.data.Dataset + tf.function + XLA 
    #This line of code enable XLA
    (accelerate Linear Algebar ) (tf.config.optizer.set_jit(True))
    5. - Using Mix-Precision for Optimize faster Through Put of reading data 
        By Get Configure in the Optimizer 
    # these serveral Line of Code for the implementation 
        5.1 Loss cale for (changing the numeric -- Not changine any mathematic equa)
        loss_cale= "dynamic
        5.2 Policy 
        policy =tf.keras.mix_precision.experimental.Policy("mix_float16", loss_scale=loss_scale)
        tf.keras.mix_precision.experimental.set_policy(policy)

        optimizer= tf.keras.mix_precision.experimental.LossScaleOptimizer(optimizer, loss_scale=loss_scale)

'''

import timeit
import tensorflow as tf
import os
import sys
import numpy as np
from tensorflow.python.ops.gen_math_ops import mul
import json

################################################################################
'''Section for building model and test training'''
################################################################################


def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # You need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000)
    return train_dataset


def dataset_fn(global_batch_size, input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = mnist_dataset(batch_size)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    return dataset


def build_cnn_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])


# # ## ManageGPUs, Reset Configure variable ,
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
# os.environ.pop('TF_CONFIG', None)
# if '.' not in sys.path:
#     sys.path.insert(0, '.')


'''
There are two strategy of Multi-worker training (Syncronously vs Asyncronously)
## Documentation for Distributed Training Tensorflow 

2. multiworker_strategy training (Three Level Optimization)
    # https://www.youtube.com/watch?v=6ovfZW8pepo
    1: Tune NCCL to fully use the cross-host network.
    (tensor 2 release 2020) --> NCCL_SOCKET_NTHREADS manually setting 
    runing experiment to find the optimal -(recommendation setting is 8)


    2: Gradient aggregation in float16(Mix precision training 16-- 32float )
    + Setting this in Custom training loop for faster compute and aggreate gradient
    also configure in custom training loop 

    3: Parallel Overlap backward path computation with gradient aggregation
    + Setting pack_size base on nccl-allreduce benchmark finding optimal pack_size
    Also Configure in the custom training loop

# The second methods just use for the references
2. multiseverparalle training (Asyncronous training)

'''

################################################################################
'''MultiWorkerstrategy training'''
################################################################################
# This talk for Implementation

# 1. tf config would create multiple workers on externel IP addresses/ports and
# 2. Set TF-Config variable on each worker accordingly
#   + Each of the machine has different Roles
#   + Ps and worker define the task for each worker (check out this)

# 3. Prepare datastet distribute and sharing dataset
# 4. Model checkpoint -- Logs saving values
# 5. Consider type of training mode -- (Sync or Asyncro) training

# Here the Configure In the Chief Machine
# Machine 0
# tf_config = {
#     "cluster": {
#        'chief': ['140.115.59.130:12345']
#         'worker': ['140.115.59.131:12345', '140.115.59.132:12345']
#     },
#     # task provide information of current task
#     # # different worker #different task defined
#     # chief define at index 0 worker define to saving checkpoint and logs training
#     "task": {'type': 'chief', 'index': 0}
# }

# Configure in Other worker (mean other machines)
# machine 1
# tf_config is the same
#     "task": {'type': 'worker', 'index': 0}
# }
# machine 2
# tf_config is the same
#     "task": {'type': 'worker', 'index': 1}
# }


communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)
#strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with strategy.scope():

    # 3 Auto-Shard data Across Workers
    def main():
        per_worker_batch_size = 200
        num_workers = 2  # len(tf_config['cluster']['worker'])
        global_batch_size = per_worker_batch_size * num_workers

        multi_worker_dataset = strategy.distribute_datasets_from_function(
            lambda input_context: dataset_fn(global_batch_size, input_context))

        model = build_cnn_model()

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        @tf.function
        def train_step(iterator):
            """Training step function."""
            def step_fn(inputs):
                # Per-Replica step function
                x, y = inputs
                with tf.GradientTape() as tape:
                    predictions = model(x, training=True)

                    per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                   reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
                    loss = tf.nn.compute_average_loss(
                        per_batch_loss, global_batch_size=global_batch_size)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                train_accuracy.update_state(y, predictions)
                return loss
            per_replica_losses = strategy.run(step_fn, args=(next(iterator), ))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        # Checkpoint saving and Restoring weights Not whole model
        from multiprocessing import util
        # temperal_checkpoint dir
        checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

        def chief_worker(task_type, task_id):
            return task_type is None or task_type == 'chief' or (task_type == 'worker' and task_id == 0)

        def _get_temp_dir(dirpath, task_id):

            base_dirpath = 'workertemp_' + str(task_id)
            # Note future will just define our custom saving dir
            temp_dir = os.path.join(dirpath, base_dirpath)
            tf.io.gfile.makedirs(temp_dir)
            return temp_dir

        def write_filepath(filepath, task_type, task_id):
            dirpath = os.path.dirname(filepath)

            base = os.path.basename(filepath)

            if not chief_worker(task_type, task_id):
                dirpath = _get_temp_dir(dirpath, task_id)

            return os.path.join(dirpath, base)

        # Saving the variable with tf.train.Checkpoint & tf.train.CheckpointManager
        epoch = tf.Variable(initial_value=tf.constant(
            0, dtype=tf.dtypes.int64), name='epoch')

        step_in_epoch = tf.Variable(initial_value=tf.constant(
            0, dtype=tf.dtypes.int64), name='step_in_epoch')

        task_type, task_id = (strategy.cluster_resolver.task_type,
                              strategy.cluster_resolver.task_id)

        checkpoint = tf.train.Checkpoint(
            model=model, epoch=epoch, step_in_epoch=step_in_epoch)

        write_checkpoint_dir = write_filepath(
            checkpoint_dir, task_type, task_id)

        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

        # You need to restore training
        last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        if last_checkpoint:
            checkpoint.restore(last_checkpoint)

        # Training Loop
        EPOCHS = 20
        Num_step_per_epoch = 70  # testing #len_imgs/batch_size

        while epoch.numpy() < EPOCHS:
            iterator = iter(multi_worker_dataset)
            total_loss = 0.0
            num_batches = 0

            while step_in_epoch.numpy() < Num_step_per_epoch:
                total_loss += train_step(iterator)
                num_batches += 1
                step_in_epoch.assign_add(1)

            train_loss = total_loss / num_batches
            print('Epoch: %d, accuracy: %f, training_loss: %f' %
                  (epoch.numpy(), train_accuracy.result(), train_loss))
            train_accuracy.reset_states()

            checkpoint_manager.save()

            if not chief_worker(task_type, task_id):
                tf.io.gfile.rmtree(write_checkpoint_dir)

            epoch.assign_add(1)
            step_in_epoch.assign(0)

    if __name__ == '__main__':

        start_time = timeit.default_timer()
        main()
        end_time = timeit.default_timer()

        print(f"Complete_pipeline time,{end_time -start_time}")
