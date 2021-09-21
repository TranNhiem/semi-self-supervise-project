import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
from augmentation_policy import augment_image_pretraining, augment_image_finetuning
from official.vision.image_classification.augment import RandAugment

SEED = 26
AUTO = tf.data.experimental.AUTOTUNE
IMG_SIZE = 32
RESIZ_IMG = 96
SEED = 26
SEED_1 = 42


class CIFAR100:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test,
                                       self.y_test) = tf.keras.datasets.cifar100.load_data()
        self.num_train_images, self.num_test_images = self.y_train.shape[0], self.y_test.shape[0]
        self.class_names = ['airplane', 'automobile', 'bird',
                            'cat', 'deer', 'dog', 'frog', 'horse', 'sheep', 'truck']

        # Normalize training and testing images
        self.x_train = tf.cast(self.x_train / 255., tf.float32)
        self.x_test = tf.cast(self.x_test / 255., tf.float32)

        self.y_train = tf.cast(tf.squeeze(self.y_train), tf.int32)
        self.y_test = tf.cast(tf.squeeze(self.y_test), tf.int32)

    def get_batch_pretraining(self, batch_id, batch_size):
        augmented_images_1, augmented_images_2 = [], []
        for image_id in range(batch_id*batch_size, (batch_id+1)*batch_size):
            image = self.x_train[image_id]
            augmented_images_1.append(augment_image_pretraining(image))
            augmented_images_2.append(augment_image_pretraining(image))
        x_batch_1 = tf.stack(augmented_images_1)
        x_batch_2 = tf.stack(augmented_images_2)

        return x_batch_1, x_batch_2  # (bs, 32, 32, 3), (bs, 32, 32, 3)

    def simclr_dataset(self, batch_id, GLOBAL_BATCH_SIZE):

        augmented_images = []
        for image_id in range(batch_id*GLOBAL_BATCH_SIZE, (batch_id+1)*GLOBAL_BATCH_SIZE):
            image = self.x_train[image_id]
            augmented_images.append(augment_image_pretraining(image))
            augmented_images.append(augment_image_pretraining(image))
        x_batch = tf.stack(augmented_images)

        return x_batch  # (2*bs, 32, 32, 3)

    def simclr_dataset_modify(self,  GLOBAL_BATCH_SIZE):

        augmented_images = []
        # for image_id in range(batch_id*BATCH_SIZE, (batch_id+1)*BATCH_SIZE):
        for image_id in range(50000):
            image = self.x_train[image_id]
            augmented_images.append(augment_image_pretraining(image))
            augmented_images.append(augment_image_pretraining(image))
        #x_batch = tf.stack(augmented_images)
        x_batch = np.array(augmented_images)
        print(x_batch.shape)

        train_ds = (tf.data.Dataset.from_tensor_slices(x_batch)


                    # .map(lambda x: (tf.image.resize(x, (RESIZ_IMG, RESIZ_IMG))),
                    #      num_parallel_calls=AUTO,)
                    # .map(processing_resize_image,num_parallel_calls=AUTO,)
                    # .batch(GLOBAL_BATCH_SIZE)
                    # .prefetch(AUTO)
                    )
        #train_ds = tf.data.Dataset(train_ds)
        return train_ds  # (2*bs, 32, 32, 3)

    def cifar_100_distribute_train_ds(self, BATCH_SIZE,  GLOBAL_BATCH_SIZE, resize=False, ):

        if resize:

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)

                            .shuffle(1024, seed=SEED)
                            .map(lambda x: (tf.image.resize(x, (RESIZ_IMG, RESIZ_IMG))),
                                 num_parallel_calls=AUTO,)
                            # .map(processing_resize_image,num_parallel_calls=AUTO,)
                            .map(augment_image_pretraining, num_parallel_calls=AUTO,)
                            .batch(GLOBAL_BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)

                            .shuffle(1024, seed=SEED)

                            .map(lambda x: (tf.image.resize(x, (RESIZ_IMG, RESIZ_IMG))),
                            num_parallel_calls=AUTO,
                                 )
                            # .map(processing_resize_image,num_parallel_calls=AUTO,)
                            .map(augment_image_pretraining, num_parallel_calls=AUTO,)
                            .batch(GLOBAL_BATCH_SIZE)

                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        else:
            train_ds_one = tf.data.Dataset.from_tensor_slices(self.x_train)
            train_ds_one = (
                train_ds_one.shuffle(1024, seed=SEED)
                .map(lambda x: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE))),
                     num_parallel_calls=AUTO,)
                .map(augment_image_pretraining, num_parallel_calls=AUTO,)
                .batch(GLOBAL_BATCH_SIZE)
                .prefetch(AUTO)
            )

            train_ds_two = tf.data.Dataset.from_tensor_slices(self.x_train)
            train_ds_two = (
                train_ds_two.shuffle(1024, seed=SEED)

                .map(lambda x: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE))),
                     num_parallel_calls=AUTO,
                     )
                .map(augment_image_pretraining, num_parallel_calls=AUTO,)
                .batch(GLOBAL_BATCH_SIZE)

                .prefetch(AUTO)
            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        return train_ds

    # This still bug Under Development
    def cifar_100_distribute_stack_train_ds(self, BATCH_SIZE,  GLOBAL_BATCH_SIZE, resize=False,):
        if resize:

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)

                            .shuffle(1024, seed=SEED)
                            .map(lambda x: (tf.image.resize(x, (RESIZ_IMG, RESIZ_IMG))),
                                 num_parallel_calls=AUTO,)
                            # .map(processing_resize_image,num_parallel_calls=AUTO,)
                            .map(augment_image_pretraining, num_parallel_calls=AUTO,)
                            .batch(GLOBAL_BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)

                            .shuffle(1024, seed=SEED)

                            .map(lambda x: (tf.image.resize(x, (RESIZ_IMG, RESIZ_IMG))),
                            num_parallel_calls=AUTO,
                                 )
                            # .map(processing_resize_image,num_parallel_calls=AUTO,)
                            .map(augment_image_pretraining, num_parallel_calls=AUTO,)
                            .batch(GLOBAL_BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = np.stack((train_ds_one, train_ds_two), axis=0)  # axis=1

        else:
            train_ds_one = tf.data.Dataset.from_tensor_slices(self.x_train)
            train_ds_one = (
                train_ds_one.shuffle(1024, seed=SEED)
                .map(lambda x: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE))),
                     num_parallel_calls=AUTO,)
                .map(augment_image_pretraining, num_parallel_calls=AUTO,)
                .batch(GLOBAL_BATCH_SIZE)
                .prefetch(AUTO)
            )

            train_ds_two = tf.data.Dataset.from_tensor_slices(self.x_train)
            train_ds_two = (
                train_ds_two.shuffle(1024, seed=SEED)

                .map(lambda x: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE))),
                     num_parallel_calls=AUTO,
                     )
                .map(augment_image_pretraining, num_parallel_calls=AUTO,)
                .batch(GLOBAL_BATCH_SIZE)

                .prefetch(AUTO)
            )

            train_ds = np.stack((train_ds_one, train_ds_two), axis=0)

        return train_ds

    def get_batch_finetuning(self, batch_id, batch_size):
        augmented_images = []
        for image_id in range(batch_id*batch_size, (batch_id+1)*batch_size):
            image = self.x_train[image_id]
            augmented_images.append(augment_image_finetuning(image))
        x_batch = tf.stack(augmented_images)
        y_batch = tf.slice(self.y_train, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  # (bs, 32, 32, 3), (bs)

    def get_batch_testing(self, batch_id, batch_size):
        x_batch = tf.slice(
            self.x_test, [batch_id*batch_size, 0, 0, 0], [batch_size, -1, -1, -1])
        y_batch = tf.slice(self.y_test, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  # (bs, 32, 32, 3), (bs)

    def shuffle_training_data(self):
        random_ids = tf.random.shuffle(tf.range(self.num_train_images))
        self.x_train = tf.gather(self.x_train, random_ids)
        self.y_train = tf.gather(self.y_train, random_ids)


class self_distillation_dataset():

    def __init__(self, BATCH_SIZE, IMG_SIZE, different_size=False):

        self.BATCH_SIZE = BATCH_SIZE
        self.IMG_SIZE = IMG_SIZE
        self.different_size = different_size

        if self.different_size:
            # self.IMG_SIZE_Te, self.IMG_SIZE_Stud=IMG_SIZE
            # self.IMG_SIZE= IMG_SIZE[1]
            raise NotImplementedError

        (self.x_train, self.y_train), (self.x_test,
                                       self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.num_train_images, self.num_test_images = self.y_train.shape[0], self.y_test.shape[0]

        # Normalize training and testing images
        self.x_train = tf.cast(self.x_train, tf.float32)  # / 255.
        self.x_test = tf.cast(self.x_test, tf.float32) / 255.

        self.y_train = tf.cast(tf.squeeze(self.y_train), tf.float32)
        self.y_test = tf.cast(tf.squeeze(self.y_test), tf.float32)
        # print(self.y_train[1:10])
        self.y_train = tf.keras.utils.to_categorical(
            self.y_train, num_classes=10)
        print(self.y_train[1:10])

        self.y_test = tf.keras.utils.to_categorical(
            self.y_test, num_classes=10)

        self.test_ds = (tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
                        .shuffle(self.BATCH_SIZE * 100)
                        .map(lambda x, y, : (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

    @classmethod
    def sample_beta_distribution(self, size, concentration_0=0.4, concentration_1=0.4):
        gamma_1_sample = tf.random.gamma(
            shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(
            shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    @classmethod
    def mix_up(self, ds_one, ds_two, alpha_mixup=0.4):
        # unpack two datasets
        images_one, labels_one = ds_one
        images_two, labels_two = ds_two
        batch_size = tf.shape(images_one)[0]

        # sample lambda and reshape it to do mixup
        l = self.sample_beta_distribution(
            batch_size, concentration_0=alpha_mixup, concentration_1=alpha_mixup)
        # print(l)

        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size, 1))

        # perform mixup on both images and labels pair imags/labels
        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)

        return (images, labels)

    @classmethod
    def sample_beta_distribution_cutmix(self, size, concentration_0=0.4, concentration_1=0.4):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    @classmethod
    @tf.function
    def get_box(self, IMG_SIZE, lambda_value):
        cut_rat = tf.math.sqrt(1.0 - lambda_value)

        cut_w = int(IMG_SIZE) * cut_rat  # rws
        cut_w = tf.cast(cut_w, tf.int32)

        cut_h = int(IMG_SIZE) * cut_rat  # rh
        cut_h = tf.cast(cut_h, tf.int32)

        cut_x = tf.random.uniform(
            (1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # rx
        cut_y = tf.random.uniform(
            (1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # ry

        boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
        boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
        bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
        bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

        target_h = bby2 - boundaryy1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundaryx1
        if target_w == 0:
            target_w += 1

        print(
            f'Size of BBox, and Target_size checking{boundaryx1, boundaryy1,target_h, target_w }')

        return boundaryx1, boundaryy1, target_h, target_w

    @classmethod
    @tf.function
    def cutmix(self, train_ds_one, train_ds_two, IMG_SIZE, alpha_cutmix):
        (image1, label1), (image2, label2) = train_ds_one, train_ds_two

        # alpha_cutmix = [0.3]
        # alpha_cutmix = [0.3]

        # Get a sample from the Beta distribution
        lambda_value = self.sample_beta_distribution(
            1, alpha_cutmix, alpha_cutmix)

        # Define Lambda
        lambda_value = lambda_value[0][0]

        # Get the bounding box offsets, heights and widths
        boundaryx1, boundaryy1, target_h, target_w = self.get_box(
            IMG_SIZE, lambda_value)

        # Get a patch from the second image (`image2`)
        crop2 = tf.image.crop_to_bounding_box(
            image2, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        image2 = tf.image.pad_to_bounding_box(
            crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )
        # Get a patch from the first image (`image1`)
        crop1 = tf.image.crop_to_bounding_box(
            image1, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        image1 = image1 - img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        image = image1 + image2

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)

        # Combine the labels of both images
        label = lambda_value * label1 + (1 - lambda_value) * label2

        return image, label

    def mixup_dataset(self, alpha_mixup):

        # two identical traing data for mixthem up
        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 100)
                        .map(lambda x, y: (x/255., y),
                             num_parallel_calls=AUTO,)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(
                        lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                        num_parallel_calls=AUTO,)
                        .map(lambda x, y: (x/255., y),
                             num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 200)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        '''
        new_x = lambda * x1 + (1 - lambda) * x2 (where x1 and x2 are images) 
        and the same equation is applied to the labels as well.
        '''

        train_ds_mu = train_ds.map(
            lambda ds_one, ds_two: (self.mix_up(ds_one, ds_two, alpha_mixup=alpha_mixup)), num_parallel_calls=AUTO
        )  # lambda ds_one, ds_two:(tf.py_function(

        return train_ds_mu, self.test_ds

    def RandAugment_two_trasforms(self, Student_RandAug, Teacher_RandAug):
        n_stu = int(Student_RandAug['number_transform'])
        m_stu = int(Student_RandAug['magnitude_transform'])
        n_te = int(Teacher_RandAug['number_transform'])
        m_te = int(Teacher_RandAug['magnitude_transform'])

        randAug_Te = iaa.RandAugment(n=n_te, m=m_te)
        randAug_Stu = iaa.RandAugment(n=n_stu, m=m_stu)

        def augment_Stu(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug_Stu(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        def augment_Te(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)
            images = randAug_Te(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds_Te = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                       .shuffle(1024, seed=SEED)
                       .batch(self.BATCH_SIZE)
                       .map(
            lambda x, y: (tf.image.resize(
                x, (self.IMG_SIZE, self.IMG_SIZE)), y),
            num_parallel_calls=AUTO,
        )
            .map(lambda x, y: (tf.py_function(augment_Te, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)

            .prefetch(AUTO)
        )

        train_ds_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .shuffle(1024, seed=SEED)
                        .batch(self.BATCH_SIZE)
                        .map(
            lambda x, y: (tf.image.resize(
                        x, (self.IMG_SIZE, self.IMG_SIZE)), y),
            num_parallel_calls=AUTO,
        )
            .map(lambda x, y: (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)

            .prefetch(AUTO)
        )

        train_ds = tf.data.Dataset.zip((train_ds_Stu, train_ds_Te))

        if self.different_size:
            train_ds_Te = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                           .shuffle(1024, seed=SEED)
                           .batch(self.BATCH_SIZE)
                           .map(
                lambda x, y: (tf.image.resize(
                    x, (self.IMG_SIZE_Te, self.IMG_SIZE_Te)), y),
                num_parallel_calls=AUTO,
            )
                .map(lambda x, y: (tf.py_function(augment_Te, [x], [tf.float32])[0], y),
                     num_parallel_calls=AUTO,)
                .map(lambda x, y: (x/255., y),
                     num_parallel_calls=AUTO,)
                .prefetch(AUTO)
            )

            train_ds_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .shuffle(1024, seed=SEED)
                            .batch(self.BATCH_SIZE)
                            .map(
                lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE_Stu, self.IMG_SIZE_Stu)), y),
                num_parallel_calls=AUTO,
            )
                .map(lambda x, y: (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                     num_parallel_calls=AUTO,)
                .map(lambda x, y: (x/255, y),
                     num_parallel_calls=AUTO,)
                .prefetch(AUTO)
            )

            train_ds = tf.data.Dataset.zip((train_ds_Te, train_ds_Stu))

        return train_ds, self.test_ds

    def RandAugment_single(self, RandAug_one):
        n = int(RandAug_one['number_transform'])
        m = int(RandAug_one['magnitude_transform'])

        randAug_one = iaa.RandAugment(n=n, m=m)

        def augment(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug_one(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                    .shuffle(1024, seed=SEED)
                    .batch(self.BATCH_SIZE)
                    .map(
                    lambda x, y: (tf.image.resize(
                        x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                    num_parallel_calls=AUTO,
                    )
                    .map(lambda x, y: (tf.py_function(augment, [x], [tf.float32])[0], y),
                         num_parallel_calls=AUTO,)
                    .map(lambda x, y: (x/255, y),
                         num_parallel_calls=AUTO,)

                    .prefetch(AUTO)
                    )

        return train_ds, self.test_ds

    def Randaug_Mixup_two_transforms(self,  alpha_mixup, Student_RandAug, Teacher_RandAug):

        n_stu = int(Student_RandAug['number_transform'])
        m_stu = int(Student_RandAug['magnitude_transform'])
        n_te = int(Teacher_RandAug['number_transform'])
        m_te = int(Teacher_RandAug['magnitude_transform'])

        randAug_Te = iaa.RandAugment(n=n_te, m=m_te)
        randAug_Stu = iaa.RandAugment(n=n_stu, m=m_stu)

        def augment_Stu(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug_Stu(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        def augment_Te(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)
            images = randAug_Te(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds_one_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                                 num_parallel_calls=AUTO,)
                            .shuffle(self.BATCH_SIZE*100, seed=SEED)
                            .batch(self.BATCH_SIZE)
                            .map(lambda x, y: (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                            num_parallel_calls=AUTO,)
                            .map(lambda x, y: (x/255., y),
                                 num_parallel_calls=AUTO,)
                            .prefetch(AUTO)
                            )

        train_ds_two_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .map(
            lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
            num_parallel_calls=AUTO,)
            .shuffle(self.BATCH_SIZE*100, seed=SEED_1)
            .batch(self.BATCH_SIZE)
            .map(lambda x, y: (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        train_ds_Stu = tf.data.Dataset.zip(
            (train_ds_one_Stu, train_ds_two_Stu))

        train_ds_one_Te = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                           .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                                num_parallel_calls=AUTO,)
                           .shuffle(self.BATCH_SIZE*100, seed=SEED)
                           .batch(self.BATCH_SIZE)
                           .map(lambda x, y: (tf.py_function(augment_Te, [x], [tf.float32])[0], y),
                                num_parallel_calls=AUTO,)
                           .map(lambda x, y: (x/255., y),
                                num_parallel_calls=AUTO,)
                           .prefetch(AUTO)
                           )

        train_ds_two_Te = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                           .map(
            lambda x, y: (tf.image.resize(
                x, (self.IMG_SIZE, self.IMG_SIZE)), y),
            num_parallel_calls=AUTO,)
            .shuffle(self.BATCH_SIZE * 100, seed=SEED_1)
            .batch(self.BATCH_SIZE)
            .map(lambda x, y: (tf.py_function(augment_Te, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        train_ds_Te = tf.data.Dataset.zip((train_ds_one_Te, train_ds_two_Te))

        train_ds_Rand_Mix_Stu = train_ds_Stu.map(
            lambda ds_one, ds_two: self.mix_up(ds_one, ds_two, alpha_mixup=alpha_mixup), num_parallel_calls=AUTO)

        train_ds_Rand_Mix_Te = train_ds_Te.map(
            lambda ds_one, ds_two: self.mix_up(ds_one, ds_two, alpha_mixup=alpha_mixup), num_parallel_calls=AUTO)

        if self.different_size:
            raise NotImplementedError

        train_ds_rand_mix = tf.data.Dataset.zip(
            (train_ds_Rand_Mix_Stu, train_ds_Rand_Mix_Te))
        return train_ds_rand_mix, self.test_ds

    def Randaug_Mixup_single_transform(self,  alpha_mixup, Student_RandAug):

        n_stu = int(Student_RandAug['number_transform'])
        m_stu = int(Student_RandAug['magnitude_transform'])
        print(n_stu)

        randAug_Stu = iaa.RandAugment(n=n_stu, m=m_stu)

        def augment_Stu(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug_Stu(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds_one_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                                 num_parallel_calls=AUTO,)
                            .shuffle(self.BATCH_SIZE*100, seed=SEED)
                            .batch(self.BATCH_SIZE)
                            .map(lambda x, y: (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                            num_parallel_calls=AUTO,)
                            .map(lambda x, y: (x/255., y),
                                 num_parallel_calls=AUTO,)
                            .prefetch(AUTO)
                            )

        train_ds_two_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .map(
            lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
            num_parallel_calls=AUTO,)
            .shuffle(self.BATCH_SIZE*100, seed=SEED_1)
            .batch(self.BATCH_SIZE)
            .map(lambda x, y: (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        train_ds_Stu = tf.data.Dataset.zip(
            (train_ds_one_Stu, train_ds_two_Stu))

        train_ds_Rand_Mix_Stu = train_ds_Stu.map(
            lambda ds_one, ds_two: self_distillation_dataset.mix_up(ds_one, ds_two, alpha_mixup=alpha_mixup), num_parallel_calls=AUTO)

        if self.different_size:
            raise NotImplementedError

        return train_ds_Rand_Mix_Stu, self.test_ds

    def Mixup_Randaug_single_transform(self, alpha_mixup, RandAug):

        n = int(RandAug['number_transform'])
        m = int(RandAug['magnitude_transform'])
        print(m)
        randAug = iaa.RandAugment(n=n, m=m)

        def rand_augment(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 100)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(
                        lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                        num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 100)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        train_ds_mixup_rand = (train_ds.shuffle(1024)
                               .map(lambda ds_one, ds_two: self.mix_up(ds_one, ds_two, alpha_mixup=alpha_mixup), num_parallel_calls=AUTO,)
                               # .batch(self.BATCH_SIZE)
                               .map(lambda x, y:  (tf.py_function(rand_augment, [x], [tf.float32])[0], y),
                                    num_parallel_calls=AUTO,)
                               .map(lambda x, y: (x/255., y),
                                    num_parallel_calls=AUTO,)
                               .prefetch(AUTO)
                               )

        return train_ds_mixup_rand, self.test_ds

    def Mixup_Randaug_two_transform(self, alpha_mixup, Student_RandAug, Teacher_RandAug):

        n_stu = int(Student_RandAug['number_transform'])
        m_stu = int(Student_RandAug['magnitude_transform'])
        n_te = int(Teacher_RandAug['number_transform'])
        m_te = int(Teacher_RandAug['magnitude_transform'])

        randAug_Te = iaa.RandAugment(n=n_te, m=m_te)
        randAug_Stu = iaa.RandAugment(n=n_stu, m=m_stu)

        def augment_Stu(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug_Stu(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        def augment_Te(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)
            images = randAug_Te(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 100)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(
                        lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                        num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 200)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        train_ds_mixup_rand_stu = (train_ds.shuffle(1024)
                                   .map(lambda ds_one, ds_two: self_distillation_dataset.mix_up(ds_one, ds_two, alpha_mixup=alpha_mixup), num_parallel_calls=AUTO,)
                                   # ss.batch(self.BATCH_SIZE)
                                   .map(lambda x, y:  (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                                        num_parallel_calls=AUTO,)
                                   .map(lambda x, y: (x/255., y),
                                        num_parallel_calls=AUTO,)
                                   .prefetch(AUTO)
                                   )

        train_ds_mixup_rand_te = (train_ds.shuffle(1024)
                                  .map(lambda ds_one, ds_two: self_distillation_dataset.mix_up(ds_one, ds_two, alpha_mixup=alpha_mixup), num_parallel_calls=AUTO,)
                                  # .batch(self.BATCH_SIZE)
                                  .map(lambda x, y:  (tf.py_function(augment_Te, [x], [tf.float32])[0], y),
                                       num_parallel_calls=AUTO,)
                                  .map(lambda x, y: (x/255., y),
                                       num_parallel_calls=AUTO,)
                                  .prefetch(AUTO)
                                  )
        train_ds_mixup_rand = tf.data.Dataset.zip(
            (train_ds_mixup_rand_stu, train_ds_mixup_rand_te))

        return train_ds_mixup_rand, self.test_ds

    def cutmix_dataset(self, alpha_cutmix):

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .shuffle(self.BATCH_SIZE * 100)
                        # .batch(BATCH_SIZE)

                        .map(lambda x, y: (x/255., y),
                             num_parallel_calls=AUTO,)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .shuffle(self.BATCH_SIZE * 100)
                        .map(lambda x, y: (x/255., y),
                             num_parallel_calls=AUTO,)
                        # .batch(BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        # Combine two shuffled datasets from the same training data.
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        train_ds_cmu = (
            train_ds.shuffle(1024)
            .map(lambda ds_one, ds_two: self.cutmix(ds_one, ds_two, self.IMG_SIZE, alpha_cutmix), num_parallel_calls=AUTO)
            .batch(self.BATCH_SIZE)
            .prefetch(AUTO)
        )

        return train_ds_cmu, self.test_ds,

    def Cutmix_Randaug_single_transform(self,  alpha_cutmix, RandAug):

        n = int(RandAug['number_transform'])
        m = int(RandAug['magnitude_transform'])

        randAug = iaa.RandAugment(n=n, m=m)

        def rand_augment(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 100)
                        # .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(
                        lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                        num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 100)
                        # .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        train_ds_cmu_rand = (
            train_ds.shuffle(1024)
            .map(lambda ds_one, ds_two: self.cutmix(ds_one, ds_two, self.IMG_SIZE, alpha_cutmix,), num_parallel_calls=AUTO)
            .batch(self.BATCH_SIZE)
            .map(lambda x, y:  (tf.py_function(rand_augment, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        return train_ds_cmu_rand, self.test_ds,

    def Cutmix_Randaug_two_transform(self,  alpha_cutmix, Student_RandAug, Teacher_RandAug):

        n_stu = int(Student_RandAug['number_transform'])
        m_stu = int(Student_RandAug['magnitude_transform'])
        n_te = int(Teacher_RandAug['number_transform'])
        m_te = int(Teacher_RandAug['magnitude_transform'])

        randAug_Te = iaa.RandAugment(n=n_te, m=m_te)
        randAug_Stu = iaa.RandAugment(n=n_stu, m=m_stu)

        def augment_Stu(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug_Stu(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        def augment_Te(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)
            images = randAug_Te(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 100,)
                        # .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                        .map(
                        lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                        num_parallel_calls=AUTO,)
                        .shuffle(self.BATCH_SIZE * 100, )
                        # .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        train_ds_cmu_rand_stu = (
            train_ds.shuffle(1024)
            .map(lambda ds_one, ds_two: self.cutmix(ds_one, ds_two, self.IMG_SIZE, alpha_cutmix,), num_parallel_calls=AUTO)
            .batch(self.BATCH_SIZE)
            .map(lambda x, y:  (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        train_ds_cmu_rand_te = (
            train_ds.shuffle(1024)
            .map(lambda ds_one, ds_two: self.cutmix(ds_one, ds_two, self.IMG_SIZE, alpha_cutmix, ), num_parallel_calls=AUTO)
            .batch(self.BATCH_SIZE)
            .map(lambda x, y:  (tf.py_function(augment_Te, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        train_ds_cmu_rand = tf.data.Dataset.zip(
            (train_ds_cmu_rand_stu, train_ds_cmu_rand_te))

        return train_ds_cmu_rand, self.test_ds,

    # These two method still Under experimetn

    def Randaug_Cutmix_two_transform(self, alpha_cutmix, Student_RandAug, Teacher_RandAug):
        n_stu = int(Student_RandAug['number_transform'])
        m_stu = int(Student_RandAug['magnitude_transform'])
        n_te = int(Teacher_RandAug['number_transform'])
        m_te = int(Teacher_RandAug['magnitude_transform'])

        randAug_Te = RandAugment(num_layers=n_te, magnitude=m_te)
        randAug_Stu = RandAugment(num_layers=n_stu, magnitude=m_stu)

        def augment_Stu(images, y):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            #images = tf.cast(images, tf.uint8)
            #images= tf.cast(images, tf.float32)
            images = randAug_Te.distort(images)

            return images, y

        def augment_Te(images, y):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.

            #images = tf.cast(images, tf.uint8)
            #images= tf.cast(images, tf.float32)
            images = randAug_Stu.distort(images)  # images.numpy()

            return images, y

        train_ds_one_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                                 num_parallel_calls=AUTO,)
                            .shuffle(self.BATCH_SIZE * 10, seed=SEED)
                            # .batch(self.BATCH_SIZE)
                            .map(lambda x, y: (augment_Stu(x, y)),
                                 num_parallel_calls=AUTO,)
                            # .batch(self.BATCH_SIZE)
                            .map(lambda x, y: (x/255., y),
                                 num_parallel_calls=AUTO,)
                            .prefetch(AUTO)
                            )

        train_ds_two_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .map(
            lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
            num_parallel_calls=AUTO,)
            .shuffle(self.BATCH_SIZE * 10, seed=SEED_1)
            # .batch(self.BATCH_SIZE)
            .map(lambda x, y: (augment_Stu(x, y)),
                 num_parallel_calls=AUTO,)
            # .batch(self.BATCH_SIZE)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        train_ds_Stu = tf.data.Dataset.zip(
            (train_ds_one_Stu, train_ds_two_Stu))

        train_ds_one_Te = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                           .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                                num_parallel_calls=AUTO,)
                           .shuffle(self.BATCH_SIZE * 10, seed=SEED)
                           # .batch(self.BATCH_SIZE)
                           .map(lambda x, y: (augment_Te(x, y)),
                                num_parallel_calls=AUTO,)
                           # .batch(self.BATCH_SIZE)
                           .map(lambda x, y: (x/255., y),
                                num_parallel_calls=AUTO,)
                           .prefetch(AUTO)
                           )

        train_ds_two_Te = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                           .map(
            lambda x, y: (tf.image.resize(
                x, (self.IMG_SIZE, self.IMG_SIZE)), y),
            num_parallel_calls=AUTO,)
            .shuffle(self.BATCH_SIZE * 10, seed=SEED_1)
            # .batch(self.BATCH_SIZE)
            .map(lambda x, y: (augment_Te(x, y)),
                 num_parallel_calls=AUTO,)
            # .batch(self.BATCH_SIZE)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        train_ds_Te = tf.data.Dataset.zip((train_ds_one_Te, train_ds_two_Te))

        train_ds_rand_cutmix_Stu = (train_ds_Stu.map(
            lambda ds_one, ds_two: self.cutmix(ds_one, ds_two,  self.IMG_SIZE, alpha_cutmix,), num_parallel_calls=AUTO)
            # .batch(self.BATCH_SIZE)
            .prefetch(AUTO)
        )
        train_ds_rand_cutmix_Te = (train_ds_Te.map(
            lambda ds_one, ds_two: self.cutmix(ds_one, ds_two, self.IMG_SIZE, alpha_cutmix, ), num_parallel_calls=AUTO)
            # .batch(self.BATCH_SIZE)
            .prefetch(AUTO)
        )

        train_ds = tf.data.Dataset.zip(
            (train_ds_rand_cutmix_Stu, train_ds_rand_cutmix_Te))

        return train_ds, self.test_ds

    def Randaug_Cutmix_one_transform(self, alpha_cutmix, Student_RandAug):
        n_stu = int(Student_RandAug['number_transform'])
        m_stu = int(Student_RandAug['magnitude_transform'])

        randAug_Stu = iaa.RandAugment(n=n_stu, m=m_stu)

        def augment_Stu(images):
            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)

            images = randAug_Stu(images=images.numpy())
            #images = (images.astype(np.float32))/255
            return images

        train_ds_one_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                                 num_parallel_calls=AUTO,)
                            .shuffle(self.BATCH_SIZE * 100)
                            .batch(self.BATCH_SIZE)
                            .map(lambda x, y: (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                            num_parallel_calls=AUTO,)
                            .map(lambda x, y: (x/255., y),
                                 num_parallel_calls=AUTO,)
                            .prefetch(AUTO)
                            )

        train_ds_two_Stu = (tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                            .map(
            lambda x, y: (tf.image.resize(
                            x, (self.IMG_SIZE, self.IMG_SIZE)), y),
            num_parallel_calls=AUTO,)
            .shuffle(self.BATCH_SIZE * 100)
            .batch(self.BATCH_SIZE)
            .map(lambda x, y: (tf.py_function(augment_Stu, [x], [tf.float32])[0], y),
                 num_parallel_calls=AUTO,)
            .map(lambda x, y: (x/255., y),
                 num_parallel_calls=AUTO,)
            .prefetch(AUTO)
        )

        train_ds_Stu = tf.data.Dataset.zip(
            (train_ds_one_Stu, train_ds_two_Stu))

        train_ds_rand_cutmix_Stu = (train_ds_Stu.map(
            lambda ds_one, ds_two: self.cutmix(ds_one, ds_two, self.IMG_SIZE, alpha_cutmix,), num_parallel_calls=AUTO)
            .prefetch(AUTO))

        return train_ds_rand_cutmix_Stu, self.test_ds
    '''Attention Calling These Two Methods'''

    def Hill_Climbing_Aug(self, BATCH_SIZE, Student_HillAug, Teacher_HillAug):

        raise NotImplementedError
