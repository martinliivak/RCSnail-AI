import os
import glob
import logging
import traceback
import asyncio
import signal
import numpy as np
import zmq
from zmq.asyncio import Context

from commons.common_zmq import recv_array_with_json, initialize_subscriber, initialize_publisher
from commons.configuration_manager import ConfigurationManager

from src.learning.model_wrapper import ModelWrapper
from src.learning.training.generator import Generator
from utilities.transformer import Transformer
from src.utilities.recorder import Recorder


async def main_dagger(context: Context):
    config_manager = ConfigurationManager()
    conf = config_manager.config
    transformer = Transformer(conf)
    recorder = Recorder(conf, transformer)

    data_queue = context.socket(zmq.SUB)
    controls_queue = context.socket(zmq.PUB)

    try:
        # TODO some better way to handle this
        model_file = 'model_n{}_m{}_1'.format(conf.m_length, conf.m_interval)
        model_file = None
        model = ModelWrapper(conf, model_file=model_file)
        mem_slice_frames = []
        mem_slice_numerics = []
        data_count = 0
        dagger_iteration = 0

        await initialize_subscriber(data_queue, conf.data_queue_port)
        await initialize_publisher(controls_queue, conf.controls_queue_port)

        while True:
            frame, data = await recv_array_with_json(queue=data_queue)
            # TODO handle case if expert data is not available, i.e full model control
            telemetry, expert_action = data
            if frame is None or telemetry is None or expert_action is None:
                print("None datas")
                continue

            recorder.record_full(frame, telemetry, expert_action)

            mem_frame = transformer.session_frame(frame, mem_slice_frames)
            mem_telemetry = transformer.session_numeric_input(telemetry, mem_slice_numerics)
            mem_expert_action = transformer.session_expert_action(expert_action)
            if mem_frame is None or mem_telemetry is None:
                continue

            data_count += recorder.record_session(mem_frame, mem_telemetry, mem_expert_action)
            if conf.dagger_training_enabled and data_count % 1000 == 0:
                recorder.store_session_batch(1000)

            if conf.dagger_training_enabled and data_count % conf.dagger_epoch_size == 0 and dagger_iteration < conf.dagger_epochs_count:
                await fit_model_with_generator(model, conf)
                dagger_iteration += 1

            try:
                if conf.control_mode == 'full_model':
                    prediction = model.predict(mem_frame, mem_telemetry).to_dict()
                    # TODO when more aspects are predicted remove these
                    prediction['d_gear'] = mem_expert_action[0]
                    prediction['d_throttle'] = mem_expert_action[2]
                elif conf.control_mode == 'shared':
                    expert_probability = np.exp(-0.15 * dagger_iteration)
                    model_probability = np.random.random()

                    if expert_probability > model_probability:
                        prediction = expert_action
                    else:
                        prediction = model.predict(mem_frame, mem_telemetry).to_dict()
                elif conf.control_mode == 'full_expert':
                    prediction = expert_action
                else:
                    raise ValueError

                controls_queue.send_json(prediction)
                #recorder.record_post_mortem(telemetry, expert_actions, prediction)
            except Exception as ex:
                print("Predicting exception: {}".format(ex))
                traceback.print_tb(ex.__traceback__)
    except Exception as ex:
        print("Exception: {}".format(ex))
        traceback.print_tb(ex.__traceback__)
    finally:
        data_queue.close()
        controls_queue.close()

        files = glob.glob(conf.path_to_session_files + '*')
        for f in files:
            os.remove(f)
        logging.info("Session partials deleted successfully.")

        if recorder is not None:
            recorder.save_session_with_expert()


async def fit_model_with_generator(model, conf):
    logging.info("Fitting with generator")
    try:
        generator = Generator(conf, memory_tuple=(conf.m_length, conf.m_interval), batch_size=32, column_mode='steer')
        model.fit(generator)
        logging.info("Fitting done")
    except Exception as ex:
        print("Fitting exception: {}".format(ex))
        traceback.print_tb(ex.__traceback__)


def cancel_tasks(loop):
    for task in asyncio.Task.all_tasks(loop):
        task.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, cancel_tasks, loop)
    loop.add_signal_handler(signal.SIGTERM, cancel_tasks, loop)

    context = zmq.asyncio.Context()
    try:
        loop.run_until_complete(main_dagger(context))
    except Exception as ex:
        logging.error("Base interruption: {}".format(ex))
        traceback.print_tb(ex.__traceback__)
    finally:
        loop.close()
        context.destroy()
