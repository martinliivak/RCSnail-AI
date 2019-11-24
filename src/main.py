import os
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pygame
import logging
from rcsnail import RCSnail

from pipeline.interceptor import MultiInterceptor
from src.pipeline.recording.recorder import Recorder
from src.utilities.configuration_manager import ConfigurationManager
from src.utilities.pygame_utils import Car, PygameRenderer


def get_training_file_name(path_to_training):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))

    return date + "_test_" + str(int(len(files_from_same_date) / 2 + 1))


def main():
    print('RCSnail manual drive demo')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    username = os.getenv('RCS_USERNAME', '')
    password = os.getenv('RCS_PASSWORD', '')
    rcs = RCSnail()
    rcs.sign_in_with_email_and_password(username, password)

    loop = asyncio.get_event_loop()
    pygame_event_queue = asyncio.Queue()
    pygame.init()
    pygame.display.set_caption("RCSnail API manual drive demo")

    config_manager = ConfigurationManager()
    config = config_manager.config
    screen = pygame.display.set_mode((config.window_width, config.window_height))

    recorder = Recorder(config)
    interceptor = MultiInterceptor(config, recorder=recorder)

    car = Car(config, update_override=interceptor.car_update_override)
    renderer = PygameRenderer(screen, car)
    interceptor.set_renderer(renderer)

    executor = ThreadPoolExecutor(max_workers=32)
    pygame_task = loop.run_in_executor(executor, renderer.pygame_event_loop, loop, pygame_event_queue)
    render_task = asyncio.ensure_future(renderer.render(rcs))
    event_task = asyncio.ensure_future(renderer.register_pygame_events(pygame_event_queue))
    queue_task = asyncio.ensure_future(rcs.enqueue(loop, interceptor.intercept_frame, interceptor.intercept_telemetry))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Closing due to keyboard interrupt.")
    finally:
        print("Closing shop.")
        queue_task.cancel()
        pygame_task.cancel()
        render_task.cancel()
        event_task.cancel()
        pygame.quit()
        asyncio.ensure_future(rcs.close_client_session())
        interceptor.close()
        if recorder is not None:
            recorder.save_session()

        print("Shop closed.")


if __name__ == "__main__":
    main()
    sleep(1)
