
import cv2


class TrainingRecorder:
    def __init__(self, filename, resolution=(60, 40)):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(filename + ".avi", fourcc, 20.0, resolution)

    def record(self, frame, telemetry):
        print("recorder:")
        print(telemetry)
        print(frame.shape)
        self.out.write(frame.to_image())

    def stop_recording(self):
        self.out.release()

