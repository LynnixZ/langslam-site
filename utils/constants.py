from minigrid.core.actions import Actions

IDX_TO_OBJECT = {
    0: 'unseen', 1: 'empty', 2: 'wall', 3: 'floor', 4: 'door',
    5: 'key', 6: 'ball', 7: 'box', 8: 'flag', 9: 'lava', 10: 'agent',
}

IDX_TO_COLOR = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'yellow', 5: 'grey'}

IDX_TO_STATE = {0: 'open', 1: 'closed', 2: 'locked'}

COMMAND_TO_ACTION = {
    "left": Actions.left,
    "right": Actions.right,
    "forward": Actions.forward,
    "pickup": Actions.pickup,
    "drop": Actions.drop,
    "toggle": Actions.toggle,
    "done": Actions.done,
    # “turnaround”是两次 left，这个在 playthrough 里展开即可
}
