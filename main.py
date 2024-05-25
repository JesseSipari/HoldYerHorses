import cv2
import numpy as np
import mss
import pygetwindow as gw
from detect_cards import detect_cards
from LearningBot import LearningBot
import pytesseract
import os

# Configure pytesseract
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary
if not os.path.exists(tesseract_path):
    raise FileNotFoundError(f"Tesseract executable not found at {tesseract_path}")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

def detect_player_actions(frame):
    # Example: Detecting actions using OCR
    action_area = frame[800:900, 100:400]  # Adjust coordinates to focus on the relevant area
    gray_frame = cv2.cvtColor(action_area, cv2.COLOR_BGR2GRAY)
    actions_text = pytesseract.image_to_string(gray_frame, config='--psm 6')
    print(f"Detected actions: {actions_text}")
    return actions_text

def detect_chips_and_pot(frame):
    # Example: Detecting chip counts and pot size using OCR
    chip_area = frame[100:200, 500:800]  # Adjust coordinates to focus on the relevant area
    gray_frame = cv2.cvtColor(chip_area, cv2.COLOR_BGR2GRAY)
    chips_text = pytesseract.image_to_string(gray_frame, config='--psm 6')
    print(f"Detected chips and pot: {chips_text}")
    return chips_text

def detect_game_phase(frame):
    # Convert frame to grayscale for better OCR accuracy
    phase_area = frame[0:100, 0:400]  # Adjust coordinates to focus on the relevant area
    gray_frame = cv2.cvtColor(phase_area, cv2.COLOR_BGR2GRAY)
    
    # Use OCR to extract text from the frame
    phase_text = pytesseract.image_to_string(gray_frame, config='--psm 6')
    print(f"Detected phase text: {phase_text}")
    
    # Determine the game phase based on extracted text
    if "pre-flop" in phase_text.lower():
        return "Pre-flop"
    elif "flop" in phase_text.lower():
        return "Flop"
    elif "turn" in phase_text.lower():
        return "Turn"
    elif "river" in phase_text.lower():
        return "River"
    elif "showdown" in phase_text.lower():
        return "Showdown"
    else:
        return "Unknown"

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
            print(f"Detected cards: {cards}")
        except ValueError as e:
            print(f"Error detecting cards: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

        # Detect player actions
        try:
            actions = detect_player_actions(frame)
        except Exception as e:
            print(f"Error detecting player actions: {e}")
            continue

        # Detect chips and pot size
        try:
            chips_and_pot = detect_chips_and_pot(frame)
        except Exception as e:
            print(f"Error detecting chips and pot: {e}")
            continue

        # Detect the current game phase
        try:
            game_phase = detect_game_phase(frame)
            print(f"Current game phase: {game_phase}")
        except Exception as e:
            print(f"Error detecting game phase: {e}")
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
