import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
# Use ONLY tf_keras for all Keras components
import tf_keras as keras
from tf_keras import layers
from tf_keras.layers import TextVectorization as Tv
# Define constants
NUM_CHAR_TOKENS = 71
output_seq_char_len = 290

char_vectorizer = Tv(max_tokens=NUM_CHAR_TOKENS,
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation",
                                    name="char_vectorizer")
def split_chars(text):
  return " ".join(list(text))
train_df = pd.read_csv('train_df.csv')
train_sentences = list(train_df['text'])
train_chars = [split_chars(sentence) for sentence in train_sentences]
char_vectorizer.adapt(train_chars)

# from tensorflow.keras import layers
char_embed = layers.Embedding(input_dim=NUM_CHAR_TOKENS, # number of different characters
                              output_dim=25, # embedding dimension of each character (same as Figure 1 in https://arxiv.org/pdf/1612.05251.pdf)
                              mask_zero=False, # don't use masks (this messes up model_5 if set to True)
                              name="char_embed")
use_url = "use_model"
embed = hub.KerasLayer(use_url, trainable=False)



# --- STEP 1: Define the custom layer again ---
# (Must be identical to the one used during training)
class VectorizerLayer(layers.Layer):
    def __init__(self, vectorizer, **kwargs):
        super().__init__(**kwargs)
        self._vectorizer = vectorizer
    def call(self, inputs):
        return self._vectorizer(inputs)
    def get_config(self):
        return super().get_config()

# --- STEP 2: Re-build the architecture ---
# Note: You must ensure 'char_vectorizer' and 'embed' (hub layer)
# are defined/loaded in this script before running this.
def build_inference_model():
    # 1. Define internal layers here so they are 'owned' by this model context
    # Use the char_vectorizer you adapted earlier
    
    # Define Embedding inside the function
    local_char_embed = layers.Embedding(
        input_dim=NUM_CHAR_TOKENS, 
        output_dim=25, 
        mask_zero=False, 
        name="char_embed" # Must match the name in your .h5 file
    )

    # Token branch
    token_inputs = keras.Input(shape=[], dtype=tf.string, name="token_input")
    token_embeddings = hub.KerasLayer(use_url, trainable=False)(token_inputs)
    token_output = layers.Dense(128, activation="relu")(token_embeddings)

    # Char branch
    char_inputs = keras.Input(shape=(1,), dtype=tf.string, name="char_input")
    char_vectors = VectorizerLayer(char_vectorizer)(char_inputs)
    char_embeddings = local_char_embed(char_vectors) # Use the local layer
    char_bi_lstm = layers.Bidirectional(layers.LSTM(32))(char_embeddings)

    # Positional branches
    line_num_in = keras.Input(shape=(15,), dtype=tf.int32, name="line_number_input")
    line_num_out = layers.Dense(32, activation="relu")(line_num_in)

    total_lines_in = keras.Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
    total_lines_out = layers.Dense(32, activation="relu")(total_lines_in)

    # Combine
    combined = layers.Concatenate()([token_output, char_bi_lstm])
    z = layers.Dense(256, activation="relu")(combined)
    z = layers.Dropout(0.5)(z)

    final_combined = layers.Concatenate()([line_num_out, total_lines_out, z])
    output = layers.Dense(5, activation="softmax")(final_combined)

    return keras.Model(inputs=[line_num_in, total_lines_in, token_inputs, char_inputs],
                       outputs=output)

# Now Create and Load
loaded_model = build_inference_model()

# IMPORTANT: Run a 'fake' prediction or call build() to initialize the weights slots
loaded_model.build([
    (None, 15), 
    (None, 20), 
    (None,), 
    (None, 1)
])

loaded_model.load_weights("my_weights.weights.h5")
print("Model loaded successfully!")