import click
import py_cui

from framework.base import Threadable, log


class TuiThread(Threadable):
    app_name = "Poor Man's Camera Effects"

    def __init__(self):
        super().__init__()
        self.root = py_cui.PyCUI(3, 3)
        self.widget_set_A = self.root.create_new_widget_set(3,3)
        self.widget_set_A.add_button('2nd window', 1,1,command=self.open_set_B)

        self.root.apply_widget_set(self.widget_set_A)
        self.widget_set_B = self.root.create_new_widget_set(5, 5)

        self.text_box_B = self.widget_set_B.add_text_box('Test', 0, 0, column_span=2)
        self.text_box_B.add_key_command(py_cui.keys.KEY_ENTER, self.open_set_A)

    def open_set_A(self):
        self.root.apply_widget_set(self.widget_set_A)

    def open_set_B(self):
        self.root.apply_widget_set(self.widget_set_B)

    def run(self):
        self.root.start()


    def main(self):
        self.run()
        return 

        input_help = "`, G, g, i, o, h, Q"
        log("Input thread started")
        input_lock = False
        while True:
            c = click.getchar()
            if c == '`':
                input_lock = not input_lock
                log("Input Lock? {}".format(input_lock))
            else:
                if input_lock:
                    pass
                elif c == 'G':
                    self.app.cascade_detector.next_classifier('frontalface')
                elif c == 'g':
                    self.app.cascade_detector.next_classifier('profileface')
                elif c == 'i':
                    self.app.input.add_interval(-1)
                elif c == 'o':
                    self.app.input.add_interval(1)
                elif c == 'h':
                    log(input_help)
                elif c == 'Q':
                    log("Exit")
                    exit(0)
                else:
                    log(c)
