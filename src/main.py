import os
import datetime
import asyncio
import pygame
import logging
from rcsnail import RCSnail

from src.learning.model_wrapper import ModelWrapper
from src.pipeline.recording.training_recorder import TrainingRecorder
from src.utilities.pygame_utils import Car, PygameRenderer
from src.pipeline.data_interceptor import DataInterceptor

window_width = 960
window_height = 480


def get_training_file_name(path_to_training):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))

    return date + "_test_" + str(int(len(files_from_same_date) / 2 + 1))


def main():
    print('RCSnail manual drive demo')
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
    username = os.getenv('RCS_USERNAME', '')
    password = os.getenv('RCS_PASSWORD', '')

    rcs = RCSnail()
    rcs.sign_in_with_email_and_password(username, password)

    loop = asyncio.get_event_loop()
    pygame_event_queue = asyncio.Queue()
    pygame.init()
    pygame.display.set_caption("RCSnail API manual drive demo")
    screen = pygame.display.set_mode((window_width, window_height))

    # TODO refactor this into a separate configuration manager
    recording_resolution = (60, 40)
    path_to_training = "../training/"
    path_to_models = "../training/models/"
    training_files_path = path_to_training + get_training_file_name(path_to_training=path_to_training)
    # recorder is None or TrainingRecorder
    recorder = TrainingRecorder(training_files_path, resolution=recording_resolution)
    recorder = None

    wrapped_model = ModelWrapper(path_to_models=path_to_models)
    wrapped_model.load_model("2019_06_11_test_1")

    interceptor = DataInterceptor(resolution=recording_resolution, recorder=recorder, model=wrapped_model)
    # update_override is None or interceptor.car_update_override
    update_override = interceptor.car_update_override
    update_override = None

    car = Car(update_override=update_override)
    renderer = PygameRenderer(screen, car)
    interceptor.set_renderer(renderer)

    pygame_task = loop.run_in_executor(None, renderer.pygame_event_loop, loop, pygame_event_queue)
    render_task = asyncio.ensure_future(renderer.render(rcs))
    event_task = asyncio.ensure_future(renderer.register_pygame_events(pygame_event_queue))
    queue_task = asyncio.ensure_future(rcs.enqueue(loop, interceptor.intercept_frame, interceptor.intercept_telemetry))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Closing due to keyboard interrupt.")
    finally:
        queue_task.cancel()
        pygame_task.cancel()
        render_task.cancel()
        event_task.cancel()
        pygame.quit()
        asyncio.ensure_future(rcs.close_client_session())

        if recorder is not None:
            recorder.save_session()


if __name__ == "__main__":
    main()
