import os
import datetime
import asyncio
import logging
import numpy as np
import zmq
from zmq.asyncio import Context, Socket

from src.pipeline.recording.recorder import Recorder
from src.utilities.configuration_manager import ConfigurationManager


def get_training_file_name(path_to_training):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))

    return date + "_test_" + str(int(len(files_from_same_date) / 2 + 1))


async def main(context: Context):
    print("started")
    config_manager = ConfigurationManager()
    recorder = Recorder(config_manager.config)

    data_queue = await initialize_synced_sub(context)
    count = 0

    while True:
        msg = await recv_array(queue=data_queue)
        print(msg)

        count += 1
        if count > 9:
            break

    data_queue.close()
    if recorder is not None:
        recorder.save_session()


async def initialize_synced_sub(context: Context):
    queue = context.socket(zmq.SUB)
    queue.connect('tcp://localhost:5561')
    queue.setsockopt(zmq.SUBSCRIBE, b'')

    synchronizer = context.socket(zmq.REQ)
    synchronizer.connect('tcp://localhost:5562')
    synchronizer.send(b'')
    await synchronizer.recv()
    synchronizer.close()

    return queue


async def recv_array(queue: Socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    metadata = await queue.recv_json(flags=flags)
    msg = await queue.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    data = np.frombuffer(buf, dtype=metadata['dtype'])
    return data.reshape(metadata['shape'])


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
