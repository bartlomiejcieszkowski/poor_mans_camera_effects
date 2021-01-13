import abc
from collections import deque
from framework.base import log

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


class FilterManager(object):
    def __init__(self):
        self.classes = dict()
        self.list = []
        self.list.append(NoFilter())
        self.list_idx = 0

    def add(self, filter_class, name=None):
        if name is None:
            log(filter_class.__name__)
            self.classes[filter_class.__name__] = filter_class
        else:
            log(name)
            self.classes[name] = filter_class
        self.list.append(filter_class())

    def current_filter(self):
        return self.list[self.list_idx]

    def get_class(self, name):
        log(name)
        if type(name) is type:
            return self.classes.get(type(name).__name__)
        else:
            return self.classes.get(name)


class AssemblyLine(FilterBase):
    def __init__(self, filter_manager):
        super(AssemblyLine, self).__init__()
        self.filters = deque()
        self.filter_manager = filter_manager

    def process(self, frame):
        elem = frame
        for obj in self.filters:
            # log("Processing {}".format(type(obj).__name__))
            elem = obj.process(elem)
        return elem

    def add_filter(self, name):
        class_ = self.filter_manager.get_class(name)
        if class_ is None:
            log("No class {}".format(name))
            return False
        log(name)
        self.filters.append(class_())
