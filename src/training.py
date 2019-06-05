from src.learning.training.training_collector import TrainingCollector
from src.learning.training.training_file_reader import TrainingFileReader

filename = "2019_06_05_test_1"


def main():
    reader = TrainingFileReader()
    collector = TrainingCollector()

    df = reader.extract_training_telemetry(filename + ".csv")

    labels = collector.collect_labels(df)
    numeric_inputs = collector.collect_numeric_inputs(df)
    video = reader.extract_training_video(filename + ".avi")
    print(video.shape)
    print(labels.shape)
    print(numeric_inputs.shape)


if __name__ == "__main__":
    main()
