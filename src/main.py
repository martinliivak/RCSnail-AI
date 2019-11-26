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


def main(context: Context):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    loop = asyncio.get_event_loop()

    config_manager = ConfigurationManager()
    config = config_manager.config

    recorder = Recorder(config)
    interceptor = MultiInterceptor(config, recorder=recorder)

    subscriber = initialize_synced_sub(context)

    while True:
        msg = await subscriber.recv()
        if msg == b'END':
            break

    interceptor.close()
    if recorder is not None:
        recorder.save_session()


def initialize_synced_sub(context: Context):
    subscriber = context.socket(zmq.SUB)
    subscriber.connect('tcp://localhost:5561')
    # TODO add possible topics
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')
    await subscriber.recv()

    sync_client = context.socket(zmq.REQ)
    sync_client.connect('tcp://localhost:5562')
    sync_client.send(b'')
    await sync_client.recv()

    return subscriber


if __name__ == "__main__":
    context = zmq.asyncio.Context()
    try:
        main(context)
    except KeyboardInterrupt:
        print('Interrupted')
        # TODO ZMQ etc shutdown
        context.destroy()
