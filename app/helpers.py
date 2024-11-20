import os
import subprocess
import time
import uuid

import cv2
from loguru import logger
import numpy as np
import pandas as pd
import tensorflow as tf

def convert_to_mp4(video_path, output_path):
    """
    Convert a video file to MP4 format using ffmpeg.

    :param video_path: Path to the input video file.
    :param output_path: Path where the converted MP4 file will be saved.
    """
    start = time.time()
    command = f"ffmpeg -fflags +genpts -i {video_path} -r 30 {output_path}"
    subprocess.call(command, shell=True)
    logger.info(f"Conversion completed in {time.time() - start}")


def extract_audio(video_path, output_path):
    """
    Extract audio from a video file and save it to a specified path.

    :param video_path: Path to the input video file.
    :param output_path: Path where the extracted audio file will be saved.
    """
    try:
        command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
        result = subprocess.call(command, shell=True)
        if result != 0:
            logger.error(f"Audio extraction failed")
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")


def extract_frames(video_path):
    """
    Extract frames from a video file at a specified interval.

    :param video_path: Path to the input video file.
    :return: A numpy array of extracted frames resized to 150x150.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"FPS: {interval}")

    count = 0
    while count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (150, 150), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        count += interval

    frames = np.array(frames) / 255
    return frames


def read_audio_features(audio_feature_path, frames):
    """
    Read audio features from a CSV file and align them with video frames.

    :param audio_feature_path: Path to the CSV file containing audio features.
    :param frames: A numpy array of video frames.
    :return: A tuple of aligned audio features and frames.
    """
    try:
        aud = pd.read_csv(audio_feature_path, header=None)
        aud_features = np.array(aud)
        logger.info("Extracted features present")
    except Exception as e:
        aud_features = np.zeros((frames.shape[0], 68))
        logger.error("Extracted features not present")

    min_time_steps = min(frames.shape[0], aud_features.shape[0])
    aud_features = aud_features[:min_time_steps]
    frames = frames[:min_time_steps]

    return aud_features, frames


@tf.keras.utils.register_keras_serializable()
def mean_acc(y_true, y_pred):
    """
    Calculate the mean accuracy between true and predicted values.

    :param y_true: True values.
    :param y_pred: Predicted values.
    :return: Mean accuracy as a float.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    diff = tf.math.abs(y_true - y_pred)
    return tf.reduce_mean(1 - diff)


def load_model():
    """
    Load a pre-trained Keras model with a custom metric.

    :return: Loaded Keras model.
    """
    return tf.keras.models.load_model(
        "models/mod_aud_vid.h5", custom_objects={"mean_acc": mean_acc}
    )


def make_prediction(video_temp_path, audio_temp_path, audio_feature_path):
    """
    Make predictions on personality traits from video and audio features.

    :param video_temp_path: Path to the temporary video file.
    :param audio_temp_path: Path to the temporary audio file.
    :param audio_feature_path: Path to the audio features CSV file.
    :return: A dictionary of predicted OCEAN traits.
    """
    audio_feature_path = extract_audio(video_temp_path, audio_temp_path)

    res = {}
    frames = extract_frames(video_temp_path)
    aud_features, frames = read_audio_features(audio_feature_path, frames)

    model = load_model()
    prediction = model.predict([frames[np.newaxis, ...], aud_features[np.newaxis, ...]])
    prediction = prediction.tolist()[0]

    res["Openness"] = round(prediction[0] * 100)
    res["Conscientiousness"] = round(prediction[1] * 100)
    res["Extraversion"] = round(prediction[2] * 100)
    res["Agreeableness"] = round(prediction[3] * 100)
    res["Neuroticism"] = round(prediction[4] * 100)

    logger.info(f"OCEAN Traits: {res}")
    return res


def cleanup_temp_files(*file_paths):
    """
    Remove temporary files from the filesystem.

    :param file_paths: Paths to the temporary files to be removed.
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
