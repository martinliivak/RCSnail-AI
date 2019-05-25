import os
import asyncio
import pygame
import logging
from rcsnail import RCSnail

from src.pipeline.recording.training_recorder import TrainingRecorder
from src.utilities.pygame_utils import Car, PygameRenderer
from src.pipeline.data_interceptor import DataInterceptor

window_width = 960
window_height = 480


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

    interceptor = DataInterceptor()
    #car = Car(interceptor.car_update_override)
    car = Car()
    renderer = PygameRenderer(screen, car)
    recorder = TrainingRecorder("../training/tempfile")

    interceptor.set_renderer(renderer)
    interceptor.set_recorder(recorder)

    pygame_task = loop.run_in_executor(None, renderer.pygame_event_loop, loop, pygame_event_queue)
    render_task = asyncio.ensure_future(renderer.render(rcs))
    event_task = asyncio.ensure_future(renderer.register_pygame_events(pygame_event_queue))
    queue_task = asyncio.ensure_future(rcs.enqueue(loop, interceptor.intercept_frame, interceptor.intercept_telemetry))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        queue_task.cancel()
        pygame_task.cancel()
        render_task.cancel()
        event_task.cancel()
        pygame.quit()
        recorder.save_session()


if __name__ == "__main__":
    main()
