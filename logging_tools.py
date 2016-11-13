import os
from traits.api import *
from traitsui.api import *
import numpy as np
import logging
import threading
import Queue
from time import sleep
from viewers import OutputStream
import pandas as pd
from datetime import datetime
import sys

class LogView(HasTraits):
    name = Str(__name__)
    logger = Instance(logging.Logger)
    log_handler = Instance(logging.StreamHandler)
    default_format = Str('[%(levelname)s] %(asctime)s (%(threadName)-10s) : %(message)s')
    terminal = Instance(OutputStream)

    view = View(Item(name='terminal', show_label=False, style='custom'),    )

    def __init__(self,name):
        super(LogView,self).__init__()
        self.name = name

    def _terminal_default(self):
        return OutputStream()

    def setup_logger(self):
        self.logger = logging.Logger(self.name)
        self.log_handler = logging.StreamHandler(self.terminal)
        self.set_format()
        self.set_level()

    def set_format(self,fmt=None):
        if fmt is None:
            formt = self.default_format
        else:
            formt = fmt
        formatter = logging.Formatter(formt)
        self.log_handler.setFormatter(formatter)


    def set_level(self,level=logging.INFO):
        self.log_handler.setLevel(logging.INFO)