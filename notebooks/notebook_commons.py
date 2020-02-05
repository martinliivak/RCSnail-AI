import numpy as np


# axis=2 for frames, axis=0 for telems
def memory_creator(instance, memory_list, length=4, interval=2, axis=2):
    if instance is None:
        return None

    memory_list.append(instance)
    near_memory = memory_list[::-interval]

    if len(near_memory) < length:
        return None

    if len(memory_list) >= length * interval:
        memory_list.pop(0)

    return np.concatenate(near_memory, axis=axis)


def read_stored_data(reader, transformer, filename, numeric_columns, diff_columns):
    telemetry = reader.read_specific_telemetry_columns(filename + '.csv', numeric_columns)
    diffs = reader.read_specific_telemetry_columns(filename + '.csv', diff_columns)
    frames = reader.read_video(filename + '.avi')
    resized_frames = transformer.resize_and_normalize_video(frames)

    return resized_frames, telemetry.to_numpy(), diffs.to_numpy()


def create_memorized_dataset(frames, telemetry, diffs, length, interval):
    # final length diff is (length - 1) * interval
    mem_slice_frames = []
    mem_slice_telemetry = []

    len_diff = (length - 1) * interval
    mem_frames = np.zeros((frames.shape[0] - len_diff, *frames.shape[1:-1], frames.shape[-1] * length))
    mem_telems = np.zeros((telemetry.shape[0] - len_diff, telemetry.shape[1] * length))

    for i in range(0, frames.shape[0]):
        mem_frame = memory_creator(frames[i], mem_slice_frames, length=length, interval=interval, axis=2)
        mem_telem = memory_creator(telemetry[i], mem_slice_telemetry, length=length, interval=interval, axis=0)

        if mem_frame is not None:
            mem_frames[i - len_diff] = mem_frame
            mem_telems[i - len_diff] = mem_telem

    mem_diffs = diffs[len_diff:]

    assert mem_frames.shape[0] == mem_telems.shape[0] == mem_diffs.shape[0], "Lengths differ!"
    return mem_frames, mem_telems, mem_diffs