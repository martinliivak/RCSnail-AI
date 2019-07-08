import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.learning.model_wrapper import ModelWrapper
from src.learning.models import create_cnn, create_mlp, create_multi_model
from src.learning.training.training_collector import TrainingCollector
from src.learning.training.training_file_reader import TrainingFileReader
from src.learning.training.training_transformer import TrainingTransformer

filename = "2019_06_28_test_2"


def main():
    reader = TrainingFileReader()
    collector = TrainingCollector()
    transformer = TrainingTransformer()

    raw_df = reader.read_training_telemetry(filename + ".csv")
    telemetry_df = transformer.get_training_df(raw_df)
    video = reader.read_training_video(filename + ".avi")
    # TODO figure out telemetry NaN and 0 removal indexes for related video
    video = video[video.shape[0] - telemetry_df.shape[0]:, :]

    labels = collector.collect_labels(telemetry_df)
    numeric_inputs = collector.collect_numeric_inputs(telemetry_df)

    # temp graphing telemetry lags
    analyze_time_diffs_in_series(telemetry_df)
    analyze_time_diffs_between_series(telemetry_df)

    mlp = create_mlp(input_dim=numeric_inputs.shape[1], regress=False)
    cnn = create_cnn(input_shape=video.shape[1:], regress=False)
    concat_model = create_multi_model(mlp, cnn)

    video_train, video_test, input_train, input_test, y_train, y_test = train_test_split(
        video, numeric_inputs, labels, test_size=0.2)

    #create_model(concat_model, input_test, input_train, video, video_test, video_train, y_test, y_train)


def analyze_time_diffs_in_series(telemetry_frame):
    time_c2 = pd.concat([telemetry_frame["c2"], telemetry_frame.diff()["c2"]], axis=1)
    time_c2.columns = ["timestamps", "diffs"]
    time_c2["diffs"].plot(title="c2 times")
    plt.show()

    time_client = pd.concat([telemetry_frame["now"], telemetry_frame.diff()["now"]], axis=1)
    time_client.columns = ["timestamps", "diffs"]
    time_client["diffs"].plot(title="client times")
    plt.show()


def analyze_time_diffs_between_series(telemetry_frame):
    stuff = telemetry_frame["now"] * 1000 - telemetry_frame["c2"]
    stuff.plot(title="diffs")
    plt.show()


def create_model(concat_model, input_test, input_train, video, video_test, video_train, y_test, y_train):
    wrapped_model = ModelWrapper()
    wrapped_model.create_model(concat_model)
    wrapped_model.model.summary()
    wrapped_model.fit((video_train, input_train, y_train), (video_test, input_test, y_test), epochs=5, verbose=1)
    wrapped_model.save_model(filename)
    wrapped_model.load_model(filename)
    predictions = wrapped_model.predict(video[0], {"p": 0, "p2": 0, "c": 244593, "c2": 1560248301322, "b": 3705, "sa": 511})
    print(predictions)


if __name__ == "__main__":
    main()
