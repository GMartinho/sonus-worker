from note_seq import midi_io
import note_seq
from magenta.models.onsets_frames_transcription import train_util
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import audio_label_data_utils
import numpy as np
import tensorflow.compat.v1 as tf
from pathlib import Path
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class OnsetsFrames():
    def __init__(self, path='train'):
        """Load the Onset-Frames Model (arXiv:1710.11153 [cs.SD]) and pretrained weights from the path using tensorflow and magenta.
        Args:
            path (str, optional): The path the model weights. Defaults to 'onsets-frames'.
        """
        tf.disable_eager_execution()
        tf.disable_v2_behavior()
        tf.logging.set_verbosity(tf.logging.ERROR)

        self.config = configs.CONFIG_MAP['onsets_frames']
        self.hparams = self.config.hparams
        self.hparams.use_cudnn = False
        self.hparams.batch_size = 1
        self.checkpoint_dir = path

        self.examples = tf.placeholder(tf.string, [None])

        self.dataset = data.provide_batch(examples=self.examples, preprocess_examples=True,
                                          params=self.hparams, is_training=False, shuffle_examples=False, skip_n_initial_records=0)

        self.estimator = train_util.create_estimator(
            self.config.model_fn, self.checkpoint_dir, self.hparams)

        self.iterator = tf.data.make_initializable_iterator(self.dataset)
        self.next_record = self.iterator.get_next()

    def predict(self, path: str, wav_data=None):
        """Using the model, return the predicted note sequence of a .wav file at the given path.
        Args:
            path (str): The path to a .wav audio file. If path is "binary", then a binary must be specified.
            wav_data (bytes): The binary for the .wav file if that is easier to extract. Defaults to None whqen path is provided.
        Returns:
            NoteSequence object containing the prediction. Convertable to MIDI.
        """
        if path == "binary":
            if wav_data is None:
                raise ValueError(
                    "The binary option is chosen but a binary is not provided.")
        else:
            f = open(path, "rb")
            wav_data = f.read()
            f.close()

        ns = note_seq.NoteSequence()
        example_list = [audio_label_data_utils.create_example(
            path, ns, wav_data, velocity_range=audio_label_data_utils.velocity_range_from_sequence(ns))]
        to_process = [example_list[0].SerializeToString()]

        print('Processing complete for', path)

        sess = tf.Session()

        sess.run([
            tf.initializers.global_variables(),
            tf.initializers.local_variables()
        ])

        sess.run(self.iterator.initializer, {self.examples: to_process})

        def transcription_data(params):
            del params
            return tf.data.Dataset.from_tensors(sess.run(self.next_record))

        input_fn = infer_util.labels_to_features_wrapper(transcription_data)
        prediction_list = list(
            self.estimator.predict(
                input_fn,
                yield_single_examples=False))
        assert len(prediction_list) == 1
        sequence_prediction = note_seq.NoteSequence.FromString(
            prediction_list[0]['sequence_predictions'][0])

        return sequence_prediction


def transcribe(record_key, midi_key) -> None:
    model = OnsetsFrames()
    from time import perf_counter
    s = perf_counter()
    pred = model.predict(record_key)
    print(f"Prediction took {(perf_counter() - s)*1000 :.1f}ms.")
    midi_io.sequence_proto_to_midi_file(pred, midi_key)
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate a MIDI file if output file is provided, otherwise plays back the keyboard.")
    parser.add_argument("-i", "--infile", type=str,
                        help="file path of the input file (excluding the .wav extention)", required=True)
    parser.add_argument("-o", "--outfile", type=str,
                        help="file path of the output file (excluding the .mid extention)")
    parser.add_argument("--fps", dest="show_fps",
                        action="store_const", const=True, default=False)
    args = parser.parse_args()

    model = OnsetsFrames()
    from time import perf_counter
    if args.outfile is not None:
        s = perf_counter()
        pred = model.predict(f"{args.infile}.wav")
        print(f"Prediction took {(perf_counter() - s)*1000 :.1f}ms.")
        midi_io.sequence_proto_to_midi_file(pred, f"{args.outfile}.mid")
    else:
        s = perf_counter()
        pred = model.predict(f"{args.infile}.wav")
        print(f"Prediction took {(perf_counter() - s)*1000 :.1f}ms.")
        from render import Seq
        from cv2 import cv2
        s = Seq(pred)
        while 1:
            if args.show_fps:
                t = perf_counter()
            frame = next(s)
            if args.show_fps:
                print(f"Live: {1/(perf_counter()-t):.1f}fps")
            cv2.imshow('frame', np.flip(frame, axis=0))
            k = cv2.waitKey(25)
            if k == ord('q'):
                break