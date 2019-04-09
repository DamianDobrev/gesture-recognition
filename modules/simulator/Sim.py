from config import CONFIG
from modules.simulator.RDS import RDS


class Simulator:
    sim = None

    def __init__(self):
        # Here we define what simulator to use.
        # If later on another one is used, this is the only thing
        # that changes. P.S. RDS has to inherit some abstract method.
        self.sim = RDS()

    def perform_action(self, gesture, offset_x=0, offset_y=0):
        print('shit', CONFIG['simulator_on'])
        if CONFIG['simulator_on']:
            self.sim.do(gesture, offset_x, offset_y)
