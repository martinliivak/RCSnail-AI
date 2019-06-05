from src.learning.training.training_collector import TrainingCollector
from src.learning.training.training_file_reader import TrainingFileReader

filename = "tempfile"


def main():
    reader = TrainingFileReader()
    collector = TrainingCollector()

    df = reader.extract_training_telemetry(filename + ".csv")
    print(df.shape)

    print(df.iloc[1])
    labels = collector.collect_labels(df)
    print(labels.iloc[1])

    video = reader.extract_training_video(filename + ".avi")
    print(video.shape)


if __name__ == "__main__":
    main()
