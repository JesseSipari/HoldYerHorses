import cv2
import numpy as np
import mss
import pygetwindow as gw
from detect_cards import detect_cards
from LearningBot import LearningBot

def main():
    # Find the window with "poker" in the title
    windows = gw.getWindowsWithTitle("poker")
    if not windows:
        print("No window with 'poker' in the title found.")
        return

    poker_window = windows[0]

    # Get the window's bounding box
    monitor = {
        "top": poker_window.top,
        "left": poker_window.left,
        "width": poker_window.width,
        "height": poker_window.height
    }

    bot = LearningBot()
    sct = mss.mss()

    while True:
        # Capture the screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Detect cards in the frame
        try:
            cards, processed_frame = detect_cards(frame)
        except ValueError as e:
            print(f"Error detecting cards: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

        # Use the bot to decide the next action
        try:
            action = bot.decide_action(cards)
        except Exception as e:
            print(f"Error in bot decision making: {e}")
            continue

        # Display the processed frame with detected card information
        cv2.imshow('Frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
