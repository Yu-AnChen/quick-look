# https://philipkiely.com/code/python_watchdog
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Watcher:

    def __init__(self, directory=".", handler=FileSystemEventHandler()):
        self.observer = Observer()
        self.handler = handler
        self.directory = directory

    def run(self):
        self.observer.schedule(
            self.handler, self.directory, recursive=False)
        self.observer.start()
        print(f"\nWatcher Running in {self.directory}/\n")
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
        self.observer.join()
        print("\nWatcher Terminated\n")


class TestHandler(FileSystemEventHandler):

    def on_any_event(self, event):
        print(event) # Your code here

if __name__=="__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    w = Watcher(path, TestHandler())
    w.run()