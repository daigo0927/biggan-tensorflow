import numpy as np
import tensorflow as tf
from glob import glob
from abc import abstractmethod, ABCMeta


class Base(metaclass=ABCMeta):
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
                 resize_shape=None,
                 crop_shape=None,
                 rotate=False,
                 flip_left_right=False,
                 flip_up_down=False):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - batch_size: int for batch size
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test

        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.rotate = rotate
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down
        
        self.batch_size = batch_size
        print('Building a dataset pipeline ...')
        self._get_samples()
        print('Found {} images.'.format(len(self)))
        self._check_channel()
        self._set_num_classes()
        self._build()
        print('Done.')

    def __len__(self):
        return len(self.samples[0])

    @abstractmethod
    def _get_samples(self): ...

    def _check_channel(self):
        self.num_channels = 3

    def _set_num_classes(self):
        self.num_classes = 1

    def _set_shape(self, images, labels):
        _, h, w, ch = images.shape.as_list()
        images.set_shape((self.batch_size, h, w, ch))
        labels.set_shape((self.batch_size, ))
        return images, labels

    def _build(self):
        """
        for eager mode, like
        --------- Example ------------
        for images, labels in (instance).loader:
            out = some_model(batch)
        ---------------------
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.samples)
        self.loader = dataset.shuffle(len(self.samples))\
          .map(self.read)\
          .map(self.preprocess)\
          .prefetch(self.batch_size)\
          .repeat()\
          .batch(self.batch_size)

    def read(self, imagefile, label):
        image = tf.io.decode_image(tf.io.read_file(imagefile), channels=self.num_channels)
        image = tf.cast(image, tf.float32)
        return image, label

    def preprocess(self, image, label):
        # resize (by nearest neighbor method)
        if self.resize_shape is not None:
            th, tw = self.resize_shape
            resize_fn = tf.image.resize_with_pad
            image = resize_fn(image, th, tw)
            image.set_shape((th, tw, self.num_channels))

        # crop
        if self.crop_shape is not None:
            image = tf.image.random_crop(image, (*self.crop_shape, self.num_channels))

        # rotate
        if self.rotate:
            raise NotImplementedError('Rotation in tf-API is not implemented.')

        # flip left-right / up-down
        if self.flip_left_right:
            image = tf.image.random_flip_left_right(image)
        if self.flip_up_down:
            image = tf.image.random_flip_up_down(image)

        image = image/127.5 - 1.0
        return image, label


class DogsVsCats(Base):
    """ tf.data pipeline for kaggle Cats and Dogs dataset.
    https://www.microsoft.com/en-us/download/details.aspx?id=54765
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
                 resize_shape=(128, 128),
                 crop_shape=None,
                 rotate=False,
                 flip_left_right=False,
                 flip_up_down=False):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - batch_size: int for batch size
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        super().__init__(dataset_dir=dataset_dir,
                         train_or_test=train_or_test,
                         batch_size=batch_size,
                         resize_shape=resize_shape,
                         crop_shape=crop_shape,
                         rotate=rotate,
                         flip_left_right=flip_left_right,
                         flip_up_down=flip_up_down)

    def _set_num_classes(self):
        self.num_classes = 2

    def _get_samples(self):
        d = self.dataset_dir + '/' + self.train_or_test
        filepath_dog = d + '/dog*.jpg'
        filepath_cat = d + '/cat*.jpg'
        imagefiles_dog = glob(filepath_dog)
        imagefiles_cat = glob(filepath_cat)
        imagefiles =  imagefiles_dog + imagefiles_cat
        labels =  [0]*len(imagefiles_dog) + [1]*len(imagefiles_cat)
        self.samples = (imagefiles, labels)
        

if __name__ == '__main__':
    pipes = [DogsVsCats
    ]

    config = {'batch_size': 4,
              'resize_shape': (128, 128),
              'crop_shape': (128, 128),
              'rotate': False,
              'flip_left_right': True,
              'flip_up_down': True}

    target_dataset_dirs = 'dataset_dirs.txt'
    with open(target_dataset_dirs, 'r') as f:
        ddirs = list(map(lambda x: x.rstrip(), f.readlines()))
    
    for pipe, ddir in zip(pipes, ddirs):
        print(f'Testing {pipe.__name__} dataset pipeline')
        for train_or_test in ['train']:
            dset = pipe(dataset_dir = ddir,
                        train_or_test = train_or_test,
                        **config)
            from tqdm import tqdm
            for i, (images, labels) in enumerate(tqdm(dset.loader)):
                if i == 100:
                    break
    
    print('Completed.')
