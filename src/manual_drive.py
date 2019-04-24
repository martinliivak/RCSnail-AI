import asyncio
import os
from rcsnail import RCSnail
from src.utilities.pygame_utils import Car, PygameRenderer
import pygame
import logging

window_width = 960
window_height = 480


class Util:
    def __init__(self, renderer):
        self.renderer = renderer
        self.frame = None

    def intercept_frame(self, frame):
        print("yo")
        print(frame)
        self.renderer.handle_new_frame(frame)
        self.frame = frame


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

    car = Car()
    renderer = PygameRenderer()
    util = Util(renderer)

    pygame_task = loop.run_in_executor(None, renderer.pygame_event_loop, loop, pygame_event_queue)
    render_task = asyncio.ensure_future(renderer.render(screen, car, rcs))
    event_task = asyncio.ensure_future(renderer.handle_pygame_events(pygame_event_queue, car))
    #queue_task = asyncio.ensure_future(rcs.enqueue(loop, renderer.handle_new_frame))
    queue_task = asyncio.ensure_future(rcs.enqueue(loop, util.intercept_frame))
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


if __name__ == "__main__":
    main()
