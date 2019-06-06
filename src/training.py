from src.learning.models import create_cnn, create_mlp
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

    # create the MLP and CNN models
    mlp = create_mlp(numeric_inputs.shape[1], regress=False)
    cnn = create_cnn(regress=False)


if __name__ == "__main__":
    main()
