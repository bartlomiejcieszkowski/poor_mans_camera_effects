import abc
from collections import deque
from random import choice

from PySide2.QtCore import Qt, Slot
from PySide2.QtWidgets import QApplication, QVBoxLayout, QLabel, QPushButton, QWidget


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
        self.classes = dict()
        self.list = []
        self.list.append(NoFilter())
        self.list_idx = 0

    def add(self, filter_class, name=None):
        if name is None:
            self.classes[filter_class.__name__] = filter_class
        else:
            self.classes[name] = filter_class
        self.list.append(filter_class())

    def current_filter(self):
        return self.list[self.list_idx]

    def get_class(self, name):
        return self.classes.get(name)


g_filter_manager = FilterManager()



class AssemblyLine(FilterBase):
    class Widget(QWidget):
        def __init__(self):
            QWidget.__init__(self)
            #super().__init__(self)
            self.test = ["1", "Test"]
            self.button = QPushButton("Click")
            self.text = QLabel("TTTT")
            self.text.setAlignment(Qt.AlignCenter)

            self.layout = QVBoxLayout()
            self.layout.addWidget(self.text)
            self.layout.addWidget(self.button)
            self.setLayout(self.layout)

            self.button.clicked.connect(self.button_clicked)

        @Slot()
        def button_clicked(self):
            self.text.setText(choice(self.test))

    def __init__(self):
        super(self).__init__()
        self.filters = deque()

    @staticmethod
    def thread_fun(this):
        this.main()

    def main(self):
        app = QApplication()
        window = AssemblyLine.Widget()
        window.resize(800, 600)
        window.show()
        app.exec_()

    def process(self, frame):
        elem = frame
        for obj in self.filters:
            elem = obj.process(elem)
        return elem

    def add_filter(self, name):
        class_ = g_filter_manager.get_class(name)
        if class_ is None:
            return False

        self.filters.append(class_())



