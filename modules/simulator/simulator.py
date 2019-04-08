import time

import pyautogui


class RDS:
    """
    This class provides a mapping of gestures and spatial values to controls
    for the RealDroneSimulator. It uses the pyautogui lib to send keyDown and
    keyUp events, effectively controlling the drone.
    https://www.realdronesimulator.com/
    """

    current_actions = None
    current_time = time.time()
    time_to_start = current_time + 5
    update_rate = 0.1  # Seconds.

    @staticmethod
    def map_gesture_and_spatial_info_to_key_event(gesture_label, is_up, is_down, is_left, is_right):
        """
        By given gesture label and a set of flags, returns a list of key event labels.
        :param gesture_label: The label of the gesture. For all possible values
            see the {RDS().do} method.
        :param is_up: True if the drone should go up.
        :param is_down: True if the drone should go down.
        :param is_left: True if the drone should roll left.
        :param is_right: True if the drone should roll right.
        :return:
        """
        if gesture_label is 'stop':
            return []

        actions = []

        if is_up:
            actions.append('w')
        elif is_down:
            actions.append('s')

        # Hover is not having left-right state.
        if gesture_label is 'hover':
            return actions

        if is_left:
            actions.append('left')
        elif is_right:
            actions.append('right')

        if gesture_label is 'left':
            actions.append('a')

        elif gesture_label is 'right':
            actions.append('d')

        elif gesture_label is 'fist' or gesture_label is 'updown':
            actions.append('up')

        elif gesture_label is 'peace':
            actions.append('r')

        elif gesture_label is 'palm':
            actions.append('down')

        return actions

    def do(self, gesture, offset_x=0, offset_y=0):
        """
        It maps the following gestures labels:
            - "stop": Does not send ANY actions.
            - "hover": Hovers the aircraft on the same place. Allows for up and down.
            - "left": Yaw to the left.
            - "right": Yaw to the right.
            - "fist" or "updown": Pitch to forth.
            - "palm": Roll back.
        An additional movement can be done based on the offset_x and offset_y,
        which is either going up or down, or roll to left or right. Offsets are
        usually the position of the hand on the screen.
        :param gesture: The gesture label. One of the specified above.
        :param offset_x: Offset along X axis. Positive offset will roll to left, negative - to right.
        :param offset_y: Offset along Y axis. Positive offset will stop the rotors, negative will
            increase the speed making the aircraft go up.
        :return:
        """
        # Game has not started YET.
        if time.time() < self.time_to_start:
            print('Game starts in... ' + str(int(self.time_to_start - time.time())) + 's')
            return

        # Make sure we only fire event if we have to!
        if time.time() - self.current_time < self.update_rate:
            return

        self.current_time = time.time()

        thresh_x = 0.6
        thresh_y = 0.2
        is_up = offset_y < -thresh_y - 0.1
        is_down = offset_y > thresh_y - 0.1
        is_right = offset_x < -thresh_x
        is_left= offset_x > thresh_x
        new_actions = RDS.map_gesture_and_spatial_info_to_key_event(gesture, is_up, is_down, is_left, is_right)

        # Beginnings are tough.
        if self.current_actions is None:
            self.current_actions = []

        # KeyUp everything that is not needed anymore...

        for action in self.current_actions:
            if action not in new_actions:
                pyautogui.keyUp(action)

        # ... and then KeyDown everything new!
        for action in new_actions:
            if action not in self.current_actions:
                pyautogui.keyDown(action)

        self.current_actions = new_actions
