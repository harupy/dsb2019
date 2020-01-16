import json


class ConfigDict(dict):
    """Configuration dictionary with convenient dot element access."""

    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    self._add(key, value)
        for key, value in kwargs.items():
            self._add(key, value)

    def _add(self, key, value):
        if isinstance(value, dict):
            self[key] = ConfigDict(value)
        else:
            self[key] = value

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(ConfigDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ConfigDict, self).__delitem__(key)
        del self.__dict__[key]

    def to_json(self):
        return json.dumps(self)

    @classmethod
    def from_json(cls, json_string):
        return cls(json.loads(json_string))
