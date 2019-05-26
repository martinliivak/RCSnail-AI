
from src.learning.training.training_reader import TrainingReader


def main():
    reader = TrainingReader()
    df = reader.extract_full_telemetry_as_dataframe("25_05_2019_test_1.csv")
    print(df.shape)

    something = reader.extract_training_video("25_05_2019_test_1.avi")
    print(something)
    print(something.shape)


if __name__ == "__main__":
    main()
