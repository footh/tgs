from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import tensorflow as tf


class Config:

    def __init__(self, file):
        assert file is not None, "File location must be provided"
        self.file = file

        with tf.gfile.Open(file, 'r') as y:
            data = y.read()
            self.data = load(data, Loader=Loader)

    def _get_nested(self, data, *keys):
        if keys and isinstance(keys, tuple):
            key = keys[0]
            if key:
                value = data.get(key)
                if len(keys) == 1:
                    return value
                else:
                    if isinstance(value, dict):
                        return self._get_nested(value, *keys[1:])

    def get(self, key):
        keys = key.split('.')
        return self._get_nested(self.data, *keys)
