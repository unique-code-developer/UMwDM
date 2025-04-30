import numpy as np

class BatchGenerator:
    def __init__(self, images, labels, batch_size=64):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.arange(len(images))
        
    def __iter__(self):
        np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.images), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_idx = self.indices[start_idx:end_idx]
            batch_images = self.images[batch_idx].astype('float32') / 255.0
            batch_labels = self.labels[batch_idx]
            yield batch_images, batch_labels