import numpy as np


class MemoryMaker:
    def __init__(self, config, memory_tuple=None):
        self.config = config
        if memory_tuple is not None:
            self.memory_length, self.memory_interval = memory_tuple
        else:
            self.memory_length = config.m_length
            self.memory_interval = config.m_interval

    # axis=2 for frames, axis=0 for telems
    def memory_creator(self, instance, memory_list, axis=2):
        if instance is None:
            return None

        memory_list.append(instance)
        near_memory = memory_list[::-self.memory_interval]

        if len(near_memory) < self.memory_length:
            return None

        if len(memory_list) >= self.memory_length * self.memory_interval:
            memory_list.pop(0)

        return np.concatenate(near_memory, axis=axis)

    def columns_from_memorized(self, memorized, columns=(1,)):
        """(1,) for steering column. (1,2,) for steering and throttle etc."""
        reshaped = memorized.reshape((self.memory_length, 4))
        return np.concatenate(reshaped[:, columns], axis=0)
