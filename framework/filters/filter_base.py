import abc


class FilterBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def process(self, frame):
        """
        This method should apply transformation on frame and return it
        :param frame:
        :return: frame type
        """
        pass

    def name(self):
        return type(self).__name__


class NoFilter(FilterBase):
    def __init__(self):
        super().__init__()

    def process(self, frame):
        return frame


class FilterManager():
    def __init__(self):
        self.list = []
        self.list.append(NoFilter())
        self.list_idx = 0

    def add(self, filter_class):
        self.list.append(filter_class())

    def current_filter(self):
        return self.list[self.list_idx]