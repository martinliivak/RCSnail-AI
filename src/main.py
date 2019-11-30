import os
import datetime
import asyncio
import logging
import zmq
from zmq.asyncio import Context, Socket

from commons.common_zmq import recv_array, initialize_synced_sub
from commons.configuration_manager import ConfigurationManager

from src.pipeline.recording.recorder import Recorder


def get_training_file_name(path_to_training):
    date = datetime.datetime.today().strftime("%Y_%m_%d")
    files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))

    return date + "_test_" + str(int(len(files_from_same_date) / 2 + 1))


async def main(context: Context):
    config_manager = ConfigurationManager()
    recorder = Recorder(config_manager.config)

    data_queue = context.socket(zmq.SUB)
    await initialize_synced_sub(context, data_queue, config_manager.config.data_queue_port)
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
