import os
import datetime
import asyncio
import logging
import signal
import zmq
from zmq.asyncio import Context, Socket

from commons.common_zmq import recv_array_with_json, initialize_synced_sub, initialize_synced_pub
from commons.configuration_manager import ConfigurationManager

from src.pipeline.recording.recorder import Recorder


def get_training_file_name(path_to_training):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))

    return date + "_test_" + str(int(len(files_from_same_date) / 2 + 1))


async def main(context: Context):
    config_manager = ConfigurationManager()
    config = config_manager.config
    recorder = Recorder(config)

    data_queue = context.socket(zmq.SUB)
    controls_queue = context.socket(zmq.PUB)

    try:
        await initialize_synced_sub(context, data_queue, config.data_queue_port)
        await initialize_synced_pub(context, controls_queue, config.controls_queue_port)

        while True:
            frame, telemetry = await recv_array_with_json(queue=data_queue)
            recorder.record(frame, telemetry)
            print(telemetry)
            print(frame.shape)
    finally:
        data_queue.close()
        #controls_queue.close()
        if recorder is not None:
            recorder.save_session()


def shutdown(loop):
    for task in asyncio.Task.all_tasks(loop):
        task.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, shutdown, loop)
    loop.add_signal_handler(signal.SIGTERM, shutdown, loop)

    context = zmq.asyncio.Context()
    try:
        loop.run_until_complete(main(context))
    except Exception as ex:
        logging.error("Interrupted base")
    finally:
        loop.close()
        context.destroy()
