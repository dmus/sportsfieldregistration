import tensorflow as tf

from sportsfieldregistration.metrics import IOUWhole
from sportsfieldregistration.utils import get_dataset
from train import loss_fun

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
tf.config.experimental.set_memory_growth(physical_devices[0], True)

metric = IOUWhole()

dataset = get_dataset('/home/derk/sports-field-registration/raw/train_val', batch_size=1)
dataset = dataset.take(1)

model = tf.keras.models.load_model('/home/derk/sports-field-registration/results/20200206000558/checkpoint-170.h5',
                                       compile=False)
model.compile(optimizer='adam', loss=loss_fun, metrics=[IOUWhole()], run_eagerly=True)

for input, target in dataset:
    prediction = model.predict(input)
    metric.get_iou(target, prediction)