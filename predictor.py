import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.layers.experimental.preprocessing import \
    StringLookup  # type: ignore


class Predictor:
    def __init__(self):
        super(Predictor, self).__init__()
        characters = [
            "]",
            '"',
            "#",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ":",
            ";",
            "?",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "[",
            "!",
        ]
        # Mapping characters to integers
        char_to_num = StringLookup(vocabulary=characters)
        # Mapping integers back to original characters
        self.num_to_char = StringLookup(
            vocabulary=char_to_num.get_vocabulary(), invert=True
        )

    def encode_single_sample(self, file_name):
        """
        Processes a single image
        """
        # 1. Read image
        img = tf.io.read_file(file_name)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [250, 600])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        return img.numpy().tolist()

    def decode_predictions(self, pred):
        """
        Greedy Decoding
        """
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = tf.keras.backend.ctc_decode(
            pred, input_length=input_len, greedy=True
        )[0][0][:, :10]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            # Hack - res+1 below due to shift in char-num mapping. [UNK] token is responsible
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def decode_predictions_beam(self, pred, beam_width=10, top_paths=1):
        """
        Greedy Decoding
        """
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results, _ = tf.keras.backend.ctc_decode(
            pred,
            input_length=input_len,
            greedy=False,
            beam_width=beam_width,
            top_paths=top_paths,
        )
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = res[:, :10]
            # Hack - res+1 below due to shift in char-num mapping. [UNK] token is responsible
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text
