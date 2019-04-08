import time

import pyautogui

current_actions = None
current_time = time.time()
time_to_start = current_time + 5
update_rate = 0.1  # Seconds.


def get_action_from_gesture(gesture_label, is_up, is_down, is_left, is_right):
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

    elif gesture_label is 'palm':
        actions.append('down')

    return actions


def do(gesture, offset_x, offset_y):
    global current_time, current_actions

    # Game has not started YET.
    if time.time() < time_to_start:
        print('Game starts in... ' + str(int(time_to_start - time.time())) + 's')
        return

    # Make sure we only fire event if we have to!
    if time.time() - current_time < update_rate:
        return

    current_time = time.time()

    thresh_x = 0.6
    thresh_y = 0.2
    is_up = offset_y < -thresh_y - 0.1
    is_down = offset_y > thresh_y - 0.1
    is_right = offset_x < -thresh_x
    is_left= offset_x > thresh_x
    new_actions = get_action_from_gesture(gesture, is_up, is_down, is_left, is_right)

    # Beginnings are tough.
    if current_actions is None:
        current_actions = []

    # KeyUp everything that is not needed anymore...

    for action in current_actions:
        if action not in new_actions:
            pyautogui.keyUp(action)

    # ... and then KeyDown everything new!
    for action in new_actions:
        if action not in current_actions:
            pyautogui.keyDown(action)

    current_actions = new_actions
