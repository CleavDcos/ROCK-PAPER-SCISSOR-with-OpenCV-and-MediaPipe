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

# Reject hand detections below this score (from MediaPipe handedness classification).
MIN_HAND_CLASSIFICATION_SCORE = 0.65


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
    """Non-thumb fingers: extended if tip is above pip (smaller y in image space)."""
    return lm[tip].y < lm[pip].y


def thumb_extended(lm, handedness_label: str) -> bool:
    """
    Thumb "up" uses horizontal spread vs IP joint (thumb differs from other fingers).
    handedness_label is MediaPipe's "Left" or "Right" for the detected hand.
    """
    if handedness_label == "Right":
        return lm[4].x < lm[3].x
    return lm[4].x > lm[3].x


def count_raised_fingers(lm, handedness_label: str) -> int:
    """
    Explicit count of raised digits (thumb + index + middle + ring + pinky).
    Used as the only signal for Rock / Paper / Scissors — no fallback shapes.
    """
    n = 0
    if thumb_extended(lm, handedness_label):
        n += 1
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if finger_up(lm, tip, pip):
            n += 1
    return n


def gesture_from_finger_count(finger_count: int):
    """
    Strict mapping only — invalid counts never map to a nearby move.

    Rock: closed fist (0 extended fingers).
    Scissors: exactly two extended fingers (typ. index + middle).
    Paper: all five extended.
    """
    if finger_count == 0:
        return "Rock"
    if finger_count == 2:
        return "Scissors"
    if finger_count == 5:
        return "Paper"
    return None


def detect_gesture(lm, handedness_label: str):
    """
    Classify a single hand from landmarks + handedness (for thumb rule).
    Returns a valid move name or None if finger count is not exactly 0, 2, or 5.
    """
    count = count_raised_fingers(lm, handedness_label)
    return gesture_from_finger_count(count)


def classify_frame_gesture(res):
    """
    Full-frame validation: multi-hand guard, confidence, then strict finger count.

    Returns:
        valid_move: "Rock" | "Paper" | "Scissors" | None (never a fake move).
        ui_primary: Short line for HUD (move, error, or invalid).
        ui_hint: Secondary line ("Show only Rock, Paper, or Scissors" when invalid).
    """
    landmarks_list = res.multi_hand_landmarks
    if not landmarks_list:
        return None, "No hand", ""

    if len(landmarks_list) > 1:
        # Deliberately do not classify when more than one hand is in frame.
        return None, "Error: Multiple hands detected", "Show only one hand"

    # Optional: filter uncertain detections using handedness classification score.
    if res.multi_handedness:
        score = res.multi_handedness[0].classification[0].score
        if score < MIN_HAND_CLASSIFICATION_SCORE:
            return None, "Uncertain detection", "Hold your hand steady"

    lm = landmarks_list[0].landmark
    label = res.multi_handedness[0].classification[0].label if res.multi_handedness else "Right"
    move = detect_gesture(lm, label)

    if move is None:
        # Any count other than 0, 2, or 5 is invalid — no nearest-gesture fallback.
        return None, "Invalid Gesture", "Show only Rock, Paper, or Scissors"

    return move, move, ""


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
    # Need >1 so we can detect two hands and refuse to classify (strict multi-hand rule).
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    user_disp = "--"
    ai_disp = "--"
    result_disp = "--"
    gesture_status = "No hand"
    gesture_hint = ""

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

        gesture, gesture_status, gesture_hint = classify_frame_gesture(res)

        if res.multi_hand_landmarks:
            if len(res.multi_hand_landmarks) > 1:
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
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

        hud_move_color = (0, 255, 0) if gesture is not None else (0, 165, 255)
        cv2.putText(frame, f"Your Move: {user_disp}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Gesture: {gesture_status}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_move_color, 2)
        if gesture_hint:
            cv2.putText(frame, gesture_hint, (10, 162),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.putText(frame, f"AI Move: {ai_disp}", (10, 192),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"Result: {result_disp}", (10, 232),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("RPS AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()