import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class BatchGenerator(Sequence):
    def __init__(self, images, labels, batch_size=64, augment=False, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_idx = self.indices[start_idx:end_idx]

        batch_images = self.images[batch_idx].astype('float32') / 255.0
        batch_labels = self.labels[batch_idx]

        if self.augment:
            batch_images = self._augment_batch(batch_images)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment_batch(self, batch):
        batch_aug = []
        for img in batch:
            img_tf = tf.convert_to_tensor(img)
            img_tf = self._augment_image(img_tf)
            batch_aug.append(img_tf.numpy())
        return np.stack(batch_aug)

    def _augment_image(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.rot90(image, k=np.random.randint(4))
        return image
