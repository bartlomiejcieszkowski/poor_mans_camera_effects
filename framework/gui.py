import os

from PySide2 import QtCore
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QGraphicsItem

from framework.base import Threadable


class UiThread(Threadable):
    class MovableBox(QGraphicsItem):
        def __init__(self, width=30, height=20):
            super(type(self), self).__init__()
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.pen_width = 1.0
            self.size = (width, height)

        def boundingRect(self) -> QtCore.QRectF:
            half_width = self.size[0] / 2
            half_height = self.size[1] / 2
            half_pen_width = self.pen_width / 2
            return QtCore.QRectF(-half_width + half_pen_width, -half_height + half_pen_width,
                                 half_width + self.pen_width, half_height + self.pen_width)

        def paint(self, painter, option, widget):
            rect = self.boundingRect()
            painter.drawRect(rect)

    class Widget(QWidget):
        def __init__(self):
            super(UiThread.Widget, self).__init__()
            # self.load_ui()
            self.create_ui()

        def load_ui(self):
            loader = QUiLoader()
            ui_file = QFile(os.path.join(os.path.dirname(__file__), "widget.ui"))
            ui_file.open(QFile.ReadOnly)
            loader.load(ui_file, self)
            ui_file.close()

        def create_ui(self):
            self.button = QPushButton("Click", parent=self)
            self.button.move(100, 100)
            self.button.show()

    @staticmethod
    def main_(this):
        this.main()

    def main(self):
        app = QApplication()
        window = UiThread.Widget()
        window.resize(800, 600)
        window.show()

        app.exec_()
        return
