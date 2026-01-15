from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time

params = BrainFlowInputParams()
# Optional: specify device name if you have multiple Muse devices
params.serial_number = "Muse-C5D9"  # Replace with your device name from the headband

board = BoardShim(BoardIds.MUSE_2_BOARD, params)
board.prepare_session()
board.start_stream()

# Collect data for 10 seconds
time.sleep(5)

# Get the data
data = board.get_board_data()
print(data.shape)
print(data)

board.stop_stream()
board.release_session()