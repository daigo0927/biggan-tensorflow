import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import xml.etree.ElementTree as ET
from glob import glob
from abc import abstractmethod, ABCMeta


class Base(metaclass=ABCMeta):
    def __init__(self,
                 dataset_dir,
                 batch_size=1,
                 transform=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory.
          - batch_size: int for batch size.
          - transform: Transform class applied to the dataset.
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

        def nofn(*x): return x
        self.transform = transform if transform is not None else nofn

        print('Building a dataset pipeline ...')
        self._get_samples()
        print('Found {} samples.'.format(len(self)))
        self._build()
        print('Done.')

    def __len__(self):
        return len(self.samples[0])

    @abstractmethod
    def _get_samples(self): ...

    def _build(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.samples)
        self.loader = dataset.shuffle(len(self.samples))\
          .map(self.read, tf.data.experimental.AUTOTUNE)\
          .map(self.transform, tf.data.experimental.AUTOTUNE)\
          .prefetch(self.batch_size)\
          .batch(self.batch_size)

        self.num_batches = int(np.ceil(len(self)/self.batch_size))

    def read(self, imagefile, label):
        image = tf.io.decode_image(tf.io.read_file(imagefile))
        return image, label


def resize_with_crop(image, target_height, target_width):
    h, w, _ = image.shape.as_list()
    r = min(h/target_height, w/target_width)
    hr, wr = int(r*target_height), int(r*target_width)
    image = tf.image.resize_with_crop_or_pad(image, hr, wr)
    image = tf.image.resize(image, (target_height, target_width))
    return image


# TODO: バウンディングボックスを切り取ってから中央を切り取る方法
class Transform:
    def __init__(self,
                 resize_shape=None,
                 crop_shape=None,
                 flip_left_right=False,
                 flip_up_down=False):
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down

    def __call__(self, image, label, bbox=None):
        image = tf.cast(image, dtype=tf.float32)
        label = tf.cast(label, dtype=tf.int32)

        if bbox is not None:
            ymin, xmin, ymax, xmax = tf.unstack(bbox)
            th, tw = ymax-ymin, xmax-xmin
            image = tf.image.crop_to_bounding_box(image, ymin, xmin, th, tw)
            print(image.shape)

        if self.resize_shape:
            image = resize_with_crop(image, *self.resize_shape)

        if self.crop_shape:
            image = tf.image.random_crop(image, (*self.crop_shape, 3))

        if self.flip_left_right:
            image = tf.image.random_flip_left_right(image)

        if self.flip_up_down:
            image = tf.image.random_flip_up_down(image)

        return image, label


class DogsVsCats(Base):
    """
    tf.data pipeline for kaggle Cats and Dogs dataset.
    https://www.microsoft.com/en-us/download/details.aspx?id=54765
    """
    def __init__(self,
                 dataset_dir,
                 batch_size=1,
                 transform=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - batch_size: int for batch size
          - transform: Transform class applied to the dataset.
        """
        super().__init__(dataset_dir=dataset_dir,
                         batch_size=batch_size,
                         transform=transform)

    def _get_samples(self):
        d = self.dataset_dir + '/' + self.train_or_test
        filepath_dog = d + '/dog*.jpg'
        filepath_cat = d + '/cat*.jpg'
        imagefiles_dog = glob(filepath_dog)
        imagefiles_cat = glob(filepath_cat)
        imagefiles =  imagefiles_dog + imagefiles_cat
        labels =  [0]*len(imagefiles_dog) + [1]*len(imagefiles_cat)
        self.samples = (imagefiles, labels)


class Cat(Base):
    """ tf.data pipeline for Cat dataset
    http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd
    thanks for https://github.com/AlexiaJM/relativistic-f-divergences
    """
    def __init__(self,
                 dataset_dir,
                 batch_size=1,
                 transform=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - batch_size: int for batch size
          - transform: Transform class applied to the dataset.
        """
        super().__init__(dataset_dir=dataset_dir,
                         batch_size=batch_size,
                         transform=transform)

    def _get_samples(self):
        filepath = self.dataset_dir + '/*.jpg'
        imagefiles = glob(filepath)
        labels = [0]*(len(imagefiles))
        self.samples = (imagefiles, labels)


def extract_boundingbox(xmlfile):
    keys = ['xmin', 'xmax', 'ymin', 'ymax', 'width', 'height']
    with tf.io.gfile.GFile(xmlfile, 'rb') as f:
        data = ET.parse(f)
        
    info = {}
    for d in data.iter():
        if d.tag.strip() in keys:
            info[d.tag.strip()] = int(d.text.strip())

    return (info['ymin'], info['xmin'], info['ymax'], info['xmax'])


class StanfordDogs(Base):
    """ tf.data pipeline for Stanford dogs dataset
    http://vision.stanford.edu/aditya86/ImageNetDogs/
    """
    def __init__(self,
                 dataset_dir,
                 batch_size=1,
                 transform=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - batch_size: int for batch size
          - transform: Transform class applied to the dataset.
        """
        super().__init__(dataset_dir=dataset_dir,
                         batch_size=batch_size,
                         transform=transform)

    def _get_samples(self):
        subdir = glob(self.dataset_dir + '/Images/*')

        imagefiles = []
        labels = []
        bboxes = []
        classes = []
        for l, d in enumerate(subdir[:1]):
            classes.append(d.split('/')[-1])
            imgfiles = glob(d+'/*.jpg')
            labs = [l]*len(imgfiles)
            annos = [f.replace('Images', 'Annotation').replace('.jpg', '') for f in imgfiles]
            bb = [extract_boundingbox(anno) for anno in annos]

            imagefiles += imgfiles
            labels += labs
            bboxes += bb

        self.samples = (imagefiles, labels, bboxes)
        self.classes = classes

    def read(self, imagefile, label, bbox):
        image = tf.io.decode_jpeg(tf.io.read_file(imagefile))
        # ymin, xmin, ymax, xmax = tf.unstack(bbox)
        # th, tw = ymax-ymin, xmax-xmin
        # image = tf.image.crop_to_bounding_box(image, ymin, xmin, th, tw)
        return image, label, bbox

            
if __name__ == '__main__':
    pipes = [DogsVsCats,
             Cat
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
