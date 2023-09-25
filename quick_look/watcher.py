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
            self.handler, self.directory, recursive=False
        )
        self.observer.start()
        print(f"\nWatcher Running in {self.directory}\n")
        try:
            while True:
                if hasattr(self.handler, 'process_events'):
                    self.handler.process_events()
                time.sleep(10)
        except Exception as e:
            print(e)
            self.observer.stop()
        self.observer.join()
        print("\nWatcher Terminated\n")


class TestHandler(FileSystemEventHandler):

    def on_any_event(self, event):
        print(event) # Your code here


class DebounceHandler(FileSystemEventHandler):
    def __init__(self):
        self.timer = None

    def on_created(self, event):
        print(event)
        if not self.timer:
            self.timer = time.time()

    def process_events(self):
        if self.timer and (time.time() - self.timer >= 1):
            self.timer = None
            self.run_task()

    def run_task(self):
        id = self.timer
        print(f'{id} run task')
        for i in range(5):
            print(f"{id} {5-i}")
            time.sleep(1)
        print(f'{id} task end')


if __name__=="__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    w = Watcher(path, DebounceHandler())
    w.run()
