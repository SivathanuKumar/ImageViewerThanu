# Utils/data_class.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the actual class definition from OCTpy
from OCTpy import Oct3Dimage

class GlobalData:
    """
    A singleton class to hold the shared application state.
    """
    def __init__(self):
        # Initialize the master image object
        self.img_obj = Oct3Dimage()

# Create a single instance of this class.
state = GlobalData()