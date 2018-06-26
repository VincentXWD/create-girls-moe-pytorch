from watchdog.observers import Observer
from watchdog.events import *
import time
import sys


class FileEventHandler(FileSystemEventHandler):
  def __init__(self):
    FileSystemEventHandler.__init__(self)

  def on_moved(self, event):
    if event.is_directory:
      print("directory moved from {0} to {1}".format(event.src_path,event.dest_path))
    else:
      print("file moved from {0} to {1}".format(event.src_path,event.dest_path))

  def on_created(self, event):
    if event.is_directory:
      print("directory created:{0}".format(event.src_path))
    else:
      print("file created:{0} sending to server".format(event.src_path))

  def on_deleted(self, event):
    if event.is_directory:
      print("directory deleted:{0}".format(event.src_path))
    else:
      print("file deleted:{0}".format(event.src_path))

  def on_modified(self, event):
    if event.is_directory:
      print("directory modified:{0}".format(event.src_path))
    else:
      print("file modified:{0}".format(event.src_path))


if __name__ == '__main__':
  observer = Observer()
  event_handler = FileEventHandler()
  path = sys.argv[1] if len(sys.argv) > 1 else '.'
  observer.schedule(event_handler, path, True)
  observer.start()
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    observer.stop()
  observer.join()