#!/usr/bin/env python3

import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import cv2
import mediapipe as mp

MOVES = ("Rock", "Paper", "Scissors")
STABLE_FRAMES = 8
ROUND_COOLDOWN = 0.7


@dataclass
class GameState:
    score_user: int = 0
    score_ai: int = 0
    difficulty: str = "Easy"

    move_history: List[str] = field(default_factory=list)
    markov: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    last_user_move_for_markov: Optional[str] = None

    stable_count: int = 0
    last_stable_gesture: Optional[str] = None

    last_round_time: float = 0.0

    show_start_screen: bool = True

    gesture_locked: bool = False
    last_played_gesture: Optional[str] = None

    # 🔥 NEW: difficulty UI feedback
    difficulty_message: str = ""
    difficulty_msg_time: float = 0.0


# ------------------ Gesture Detection ------------------ #

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def detect_gesture(lm):
    index = finger_up(lm, 8, 6)
    middle = finger_up(lm, 12, 10)
    ring = finger_up(lm, 16, 14)
    pinky = finger_up(lm, 20, 18)

    if index and middle and not ring and not pinky:
        return "Scissors"

    if index and middle and ring and pinky:
        return "Paper"

    return "Rock"


def stability(state, gesture):
    if gesture is None:
        state.stable_count = 0
        state.last_stable_gesture = None
        return None

    if gesture == state.last_stable_gesture:
        state.stable_count += 1
    else:
        state.last_stable_gesture = gesture
        state.stable_count = 1

    if state.stable_count >= STABLE_FRAMES:
        return gesture
    return None


# ------------------ AI ------------------ #

def counter(move):
    return {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}[move]


def ai_move(state):
    # EASY → Random
    if state.difficulty == "Easy":
        return random.choice(MOVES)

    # MEDIUM → Frequency counter
    if state.difficulty == "Medium":
        if not state.move_history:
            return random.choice(MOVES)
        most_common = Counter(state.move_history).most_common(1)[0][0]
        return counter(most_common)

    # HARD → Markov prediction
    prev = state.last_user_move_for_markov
    if prev and state.markov[prev]:
        predicted = max(state.markov[prev], key=state.markov[prev].get)
        return counter(predicted)

    return random.choice(MOVES)


# ------------------ MAIN ------------------ #

def main():
    state = GameState()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    user_disp = "--"
    ai_disp = "--"
    result_disp = "--"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        # ---------------- START SCREEN ---------------- #
        if state.show_start_screen:

            cv2.putText(frame, "Rock Paper Scissors AI", (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.putText(frame, f"Current Difficulty: {state.difficulty}", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            # 🔥 Difficulty change message
            if time.time() - state.difficulty_msg_time < 2:
                cv2.putText(frame, state.difficulty_message, (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame, "Press E / M / H to change difficulty", (60, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.putText(frame, "Press SPACE to Start", (120, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame, "Press Q to Quit", (160, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

            cv2.imshow("RPS AI", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('e'):
                state.difficulty = "Easy"
                state.difficulty_message = "Switched to EASY"
                state.difficulty_msg_time = time.time()

            elif key == ord('m'):
                state.difficulty = "Medium"
                state.difficulty_message = "Switched to MEDIUM"
                state.difficulty_msg_time = time.time()

            elif key == ord('h'):
                state.difficulty = "Hard"
                state.difficulty_message = "Switched to HARD"
                state.difficulty_msg_time = time.time()

            elif key == 32:  # SPACE
                state.show_start_screen = False

            continue

        # ---------------- GAME LOOP ---------------- #

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        gesture = None

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            gesture = detect_gesture(lm)
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        stable_gesture = stability(state, gesture)

        # Unlock logic
        if gesture is None or gesture != state.last_played_gesture:
            state.gesture_locked = False

        # Play round
        if stable_gesture and not state.gesture_locked and (now - state.last_round_time > ROUND_COOLDOWN):

            user = stable_gesture
            ai = ai_move(state)

            if user == ai:
                result = "Draw"
            elif (user, ai) in [("Rock","Scissors"),("Paper","Rock"),("Scissors","Paper")]:
                result = "Win"
                state.score_user += 1
            else:
                result = "Lose"
                state.score_ai += 1

            state.move_history.append(user)

            if state.last_user_move_for_markov:
                state.markov[state.last_user_move_for_markov][user] += 1
            state.last_user_move_for_markov = user

            user_disp = user
            ai_disp = ai
            result_disp = result

            state.last_round_time = now
            state.gesture_locked = True
            state.last_played_gesture = user

        # ---------------- UI ---------------- #

        cv2.putText(frame, f"Difficulty: {state.difficulty}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"User: {state.score_user}  AI: {state.score_ai}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"Your Move: {user_disp}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"AI Move: {ai_disp}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"Result: {result_disp}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("RPS AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()