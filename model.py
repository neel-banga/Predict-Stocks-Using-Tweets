# Let's create a sentiment analysys model, the input should be a string literal, and the output should be a decimal value from 0.0 (the worst) to 1.0 (the best)
# Input: "Wow I feel great" -> 0.6598318

# Import python built-in modules that are needed
import os
import string
import shutil
import re

# Import tensorflow, and other modules in tensorflow to make writing code a bit easier
import tensorflow as tf
from keras import layers, losses, Sequential
from keras.layers import TextVectorization, Dense, Dropout, Embedding, GlobalAveragePooling1D, TextVectorization


# Check tensorflow version and make sure everything is working okay
print(tf.__version__)

# Here, let's define some constant variables
BATCH_SIZE=32
SEED=42
MAX_TOKENS = 10000
OUTPUT_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16


#  Create a function to download the IMDB movie dataset
def download_dataset():
    imdb_dataset = tf.keras.utils.get_file(
        'aclImdb_v1.tar.gz', 
        'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', 
        untar=True, cache_dir='.', cache_subdir=''
    )

    imdb_dataset_dir = os.path.join(os.path.dirname(imdb_dataset), 'aclImdb')

    # Let's delete any files that we don't need and make it suitable for training
    imdb_dataset_train_dir = os.path.join(imdb_dataset_dir, 'train')
    train_dir = os.path.join(imdb_dataset_train_dir, 'unsup')
    shutil.rmtree(train_dir)



# Let's create a function to strip HTML break tags
def kill_html_tags(input_data):
    lowercase_data = tf.strings.lower(input_data)
    stripped_data = tf.strings.regex_replace(lowercase_data, '<br />', '')
    return tf.strings.regex_replace(stripped_data, '[%s]' % re.escape(string.punctuation), '')

vectorize_layer = TextVectorization(
    standardize = kill_html_tags,
    max_tokens = MAX_TOKENS,
    output_mode = 'int',
    output_sequence_length = OUTPUT_SEQUENCE_LENGTH
)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label



# Let's split our dataset, for training, validation and testiong
train_dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=BATCH_SIZE, validation_split=0.2,
    subset='training', seed=SEED
)

validation_dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=BATCH_SIZE, validation_split=0.2,
    subset='validation', seed=SEED
)

test_dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=BATCH_SIZE
)

# Adapt strings to integers
text_only_dataset = train_dataset.map(lambda x, y: x)
vectorize_layer.adapt(text_only_dataset)

training_set = train_dataset.map(vectorize_text)
validation_set = validation_dataset.map(vectorize_text)
testing_set = test_dataset.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
training_set = training_set.cache().prefetch(buffer_size=AUTOTUNE)
validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)
testing_set = testing_set.cache().prefetch(buffer_size=AUTOTUNE)



# Now let's create a function to train our raw_model
def train_model():
    # Let's create some checkpoints for our model in case our system crashes during training
    checkpoint_path = 'raw_model_1/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)


    # Now let's create our model!
    model = Sequential([
        Embedding(MAX_TOKENS + 1, EMBEDDING_DIM),
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(1)
    ])


    model.compile(
        optimizer='adam',
        loss = losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
    )

    model.fit(
        training_set,
        validation_data = validation_set,
        epochs=15,
        callbacks=[cp_callback])

    # Now, let's save our model as an h5 file
    model.save('model.h5')



# Let's load up our saved raw model
model = tf.keras.models.load_model('model.h5')


# While our model works well, it is only able to understand pre-vectorized strings, so we must create a deployment model that can process raw string literals

deployment_model = Sequential([
  vectorize_layer, 
  model, 
  layers.Activation('sigmoid')
])

deployment_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

def sentiment(text):
    return deployment_model.predict([text])
