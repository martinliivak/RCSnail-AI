from sklearn.model_selection import train_test_split

from src.learning.model_wrapper import ModelWrapper
from src.learning.models import create_cnn, create_mlp, create_multi_model
from src.learning.training.training_collector import TrainingCollector
from src.learning.training.training_file_reader import TrainingFileReader

filename = "2019_06_05_test_1"


def main():
    reader = TrainingFileReader()
    collector = TrainingCollector()

    telemetry_frame = reader.extract_training_telemetry(filename + ".csv")
    video = reader.extract_training_video(filename + ".avi")

    labels = collector.collect_labels(telemetry_frame)
    numeric_inputs = collector.collect_numeric_inputs(telemetry_frame)

    print(video.shape)
    print(labels.shape)
    print(numeric_inputs.shape)

    mlp = create_mlp(input_dim=numeric_inputs.shape[1], regress=False)
    cnn = create_cnn(input_shape=video.shape[1:], regress=False)
    concat_model = create_multi_model(mlp, cnn)

    video_train, video_test, input_train, input_test, y_train, y_test = train_test_split(
        video, numeric_inputs, labels, test_size=0.2)

    wrapped_model = ModelWrapper()
    wrapped_model.create_model(concat_model)
    wrapped_model.fit((video_train, input_train, y_train), (video_test, input_test, y_test))


if __name__ == "__main__":
    main()
