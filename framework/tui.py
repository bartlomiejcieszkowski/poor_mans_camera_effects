import click
import npyscreen

from framework.base import Threadable, log


class TuiThread(Threadable):
    app_name = "Poor Man's Camera Effects"

    class App(npyscreen.StandardApp):
        def onStart(self):
            self.addForm("MAIN", TuiThread.MainForm, TuiThread.app_name)

    class MainForm(npyscreen.ActionForm):
        def create(self):
            self.title = self.add(npyscreen.TitleText, name="Bry", value="dobry")

        def on_ok(self):
            self.parentApp.setNextform(None)

        def on_cancel(self):
            self.title.value = "dobry"

    def __init__(self, app):
        super().__init__()
        self.app = app

    def main(self):
        app = TuiThread.App()
        app.run()

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
                    self.app.camera_input.add_interval(-1)
                elif c == 'o':
                    self.app.camera_input.add_interval(1)
                elif c == 'h':
                    log(input_help)
                elif c == 'Q':
                    log("Exit")
                    exit(0)
                else:
                    log(c)
