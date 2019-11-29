import os
import datetime
import asyncio
import logging
import zmq
from zmq.asyncio import Context

from pipeline.interceptor import MultiInterceptor
from src.pipeline.recording.recorder import Recorder
from src.utilities.configuration_manager import ConfigurationManager


def get_training_file_name(path_to_training):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))

    return date + "_test_" + str(int(len(files_from_same_date) / 2 + 1))


async def main(context: Context):
    print("started")
    config_manager = ConfigurationManager()
    config = config_manager.config

    recorder = Recorder(config)
    interceptor = MultiInterceptor(config, recorder=recorder)

    subscriber = await initialize_synced_sub(context)

    while True:
        msg = await subscriber.recv()
        print(msg)
        if msg == b'END':
            break

    interceptor.close()
    if recorder is not None:
        recorder.save_session()


async def initialize_synced_sub(context: Context):
    subscriber = context.socket(zmq.SUB)
    subscriber.connect('tcp://localhost:5561')
    # TODO add possible topics
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')

    sync_client = context.socket(zmq.REQ)
    sync_client.connect('tcp://localhost:5562')
    sync_client.send(b'')
    sync_conf_msg = await sync_client.recv()
    print(sync_conf_msg)

    return subscriber


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    event_loop = asyncio.get_event_loop()

    context = zmq.asyncio.Context()
    main_task = asyncio.ensure_future(main(context), loop=event_loop)

    try:
        event_loop.run_forever()
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        # TODO figure out how to come out of the task and terminate gracefully
        context.destroy()
        main_task.cancel()
        event_loop.close()
