import traceback
import asyncio
import logging
import signal
import numpy as np
import zmq
from zmq.asyncio import Context

from commons.common_zmq import recv_array_with_json, initialize_subscriber, initialize_publisher
from commons.configuration_manager import ConfigurationManager

from src.learning.model_wrapper import ModelWrapper
from src.learning.training.training_transformer import TrainingTransformer
from src.utilities.recorder import Recorder


async def main_dagger(context: Context):
    config_manager = ConfigurationManager()
    config = config_manager.config
    recorder = Recorder(config)
    transformer = TrainingTransformer(config)

    data_queue = context.socket(zmq.SUB)
    controls_queue = context.socket(zmq.PUB)

    try:
        model = ModelWrapper(config)
        data_count = 0
        dagger_iteration = 0

        await initialize_subscriber(data_queue, config.data_queue_port)
        await initialize_publisher(controls_queue, config.controls_queue_port)

        while True:
            frame, data = await recv_array_with_json(queue=data_queue)
            telemetry, expert_actions = data

            if frame is None or telemetry is None or expert_actions is None:
                continue

            data_count += recorder.record_expert(frame, telemetry, expert_actions)

            if config.dagger_training_enabled and data_count % config.dagger_epoch_size == 0 and dagger_iteration < config.dagger_epochs_count:
                await fitting_model(model, recorder, transformer)
                dagger_iteration += 1

            try:
                if config.prediction_mode == 'full_model':
                    prediction = model.predict(frame, telemetry).to_dict()
                elif config.prediction_mode == 'shared':
                    expert_probability = np.exp(-0.15 * dagger_iteration)
                    model_probability = np.random.random()

                    if expert_probability > model_probability:
                        prediction = expert_actions
                    else:
                        prediction = model.predict(frame, telemetry).to_dict()
                elif config.prediction_mode == 'full_expert':
                    prediction = expert_actions
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

        if recorder is not None:
            recorder.save_session_with_expert()


async def fitting_model(model, recorder, transformer):
    logging.info("fitting")
    try:
        frames, telemetry, expert_actions = recorder.get_current_data()
        train, test = transformer.transform_aggregation_to_inputs(frames, telemetry, expert_actions)

        model.fit(train, test)
        logging.info("fitting done")
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
