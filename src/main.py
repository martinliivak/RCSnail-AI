import os
import time
from datetime import datetime
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

    control_mode = conf.control_mode
    dagger_training_enabled = conf.dagger_training_enabled
    dagger_epoch_size = conf.dagger_epoch_size

    try:
        model = ModelWrapper(conf, output_shape=2)
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
                logging.info("None data")
                continue

            #recorder.record_with_expert(frame, telemetry, expert_action)
            mem_frame = transformer.session_frame_wide(frame, mem_slice_frames)
            mem_telemetry = transformer.session_numeric_input(telemetry, mem_slice_numerics)
            mem_expert_action = transformer.session_expert_action(expert_action)
            if mem_frame is None or mem_telemetry is None:
                # Send back these first few instances, as the other application expects 1:1 responses
                controls_queue.send_json(expert_action)
                continue

            data_count += recorder.record_session(mem_frame, mem_telemetry, mem_expert_action)
            if control_mode == 'shared' and dagger_training_enabled and data_count % dagger_epoch_size == 0:
                recorder.store_session_batch(dagger_epoch_size)

                if dagger_iteration < conf.dagger_epochs_count:
                    # send 0 throttle so the car won't go wild during fitting
                    null_controls = expert_action.copy()
                    null_controls['d_throttle'] = 0.0
                    controls_queue.send_json(null_controls)

                    await fit_and_eval_model(model, conf)
                    dagger_iteration += 1
                    logging.info('Dagger iter {}'.format(dagger_iteration))
                    continue
                else:
                    dagger_iteration = 50
            try:
                if control_mode == 'full_expert' or expert_action['manual_override']:
                    next_controls = expert_action.copy()
                    time.sleep(0.035)
                elif control_mode == 'full_model':
                    next_controls = model.predict(mem_frame, mem_telemetry).to_dict()
                    next_controls['d_gear'] = mem_expert_action[0]
                    #next_controls['d_throttle'] = mem_expert_action[2]
                elif control_mode == 'shared':
                    expert_probability = np.exp(-0.02 * dagger_iteration)
                    model_probability = np.random.random()
                    model_action = model.predict(mem_frame, mem_telemetry).to_dict()

                    if expert_probability > model_probability:
                        next_controls = model_action
                        next_controls['d_gear'] = mem_expert_action[0]
                        next_controls['d_steering'] = mem_expert_action[1]
                        next_controls['d_throttle'] = mem_expert_action[2]
                    else:
                        next_controls = model_action
                        next_controls['d_gear'] = mem_expert_action[0]
                else:
                    raise ValueError('Misconfigured control mode!')

                recorder.record_full(frame, telemetry, expert_action, next_controls)
                controls_queue.send_json(next_controls)
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
            #recorder.save_session_with_expert()
            recorder.save_session_with_predictions()

        model.save_best_model()


async def fit_and_eval_model(model, conf):
    logging.info("Fitting with generator")
    try:
        generator = Generator(conf, batch_size=32, column_mode='steer')
        model.fit(generator, generator.generate, epochs=8, verbose=0, fresh_model=False)

        logging.info("Model evaluation")
        eval_generator = Generator(conf, eval_mode=True, batch_size=32, column_mode='steer')
        model.evaluate_model(eval_generator)

        logging.info("Best DAgger MSE: {}".format(model.min_error))
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
