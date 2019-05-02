from rl.callbacks import Callback

class ActionTrajectoryCallback(Callback):
    def __init__(self):
        self.actions = []

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        self.actions.append(logs["action"])
