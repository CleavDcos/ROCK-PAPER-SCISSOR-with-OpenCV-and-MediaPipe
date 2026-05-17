# ✊✋✌️ Rock Paper Scissors AI — OpenCV + MediaPipe

A real-time, hand-gesture-based Rock Paper Scissors game that uses your **webcam**, **OpenCV**, and **Google's MediaPipe** to detect your hand gestures and pit you against an AI opponent — with three difficulty levels, including a Markov-chain-powered Hard mode.

## 📖 Project Overview

This project implements a **computer-vision-powered Rock Paper Scissors game** where you play against an intelligent AI by showing hand gestures to your webcam. The application uses:

- **MediaPipe Hands** for accurate real-time hand landmark detection (21 landmarks per hand)
- **Strict finger-counting logic** for robust gesture classification
- **Multiple AI difficulty levels** with increasing sophistication
- **Markov chain prediction** to intelligently counter your gameplay patterns
- **OpenCV** for live video processing and interactive HUD display

The game is fully **non-networked** and runs locally on your machine, with all AI logic and hand tracking happening in real-time.

---

## 🎮 Core Features

### Hand Detection & Gesture Recognition
- **Real-time hand tracking** using MediaPipe's hand landmark detection model
- **Finger counting algorithm** that counts raised digits (thumb + 4 fingers) to classify gestures
- **Strict gesture validation**: Only accepts exactly 0 fingers (Rock), 2 fingers (Scissors), or 5 fingers (Paper)
- **Handedness awareness** — different thumb detection logic for left vs. right hands
- **Multi-hand guard** — rejects frames where more than one hand is detected to prevent ambiguous input
- **Gesture stability check** — requires 8 consecutive frames of the same gesture before registering a play (prevents false positives from jitter)
- **Confidence filtering** — rejects uncertain hand detections below a 0.65 classification confidence threshold

### AI Difficulty Levels
The game offers three distinct AI strategies that scale in complexity:

1. **🟢 Easy Mode**
   - AI picks a completely random move every round
   - Pure luck-based gameplay
   - Best for beginners or casual play

2. **🟡 Medium Mode**
   - AI analyzes your historical move frequency
   - Uses the **counter() function** to predict your most common move and counters it
   - Builds your move history over time for adaptive play
   - Provides a moderate challenge

3. **🔴 Hard Mode** (Markov Chain Prediction)
   - AI uses a **Markov chain model** to predict your *next* move based on sequences of your past moves
   - Learns state transitions: if you played Rock, what did you play next?
   - Dynamically builds a transition probability matrix (`markov` dictionary)
   - Predicts your next move and counter it with the appropriate gesture
   - Most challenging mode — requires strategic thinking to outsmart the AI

### Game Interface & Visual Feedback
- **Live HUD overlay** displaying:
  - Current scores (user vs. AI)
  - Active difficulty level
  - Detected gesture and validation status
  - AI's current move
  - Round result (Win/Lose/Draw)
- **Start screen** with difficulty selection and real-time feedback
- **Hand skeleton visualization** — draws 21 hand landmarks and finger connections on the video feed
- **Color-coded status messages**:
  - Valid gestures shown in green
  - Errors and invalid gestures shown in red
  - Helpful hints guide the user to correct their gesture

### Gameplay Mechanics
- **Stable gesture detection** — must hold your gesture steady for 8 frames before it registers (prevents accidental moves)
- **Round cooldown** — 0.7 second delay between rounds prevents immediate re-triggering
- **Round-based scoring** — scores increment only when a valid gesture is detected and classified
- **Locked gesture system** — once a gesture is played, it's locked until the next round begins
- **Difficulty toggling** — use **'U'** and **'D'** keys to cycle through difficulty levels during gameplay

---

## 🖥️ System Requirements

### Hardware
- **Webcam** — any USB or integrated camera (720p or higher recommended)
- **Processor** — modern CPU (Intel i5+ or equivalent) for real-time ML inference
- **RAM** — 4GB minimum (8GB recommended)

### Software
- **Python 3.8+** (3.10+ recommended for best performance)
- **Operating System** — Windows, macOS, or Linux

### Python Dependencies

| Library | Version | Purpose |
|---|---|---|
| `opencv-python` | 4.5+ | Real-time webcam capture, frame processing, and visual rendering |
| `mediapipe` | 0.10+ | Hand landmark detection using the Tasks API |
| `numpy` | 1.19+ | Numerical operations (bundled with MediaPipe) |

---

## 🚀 Installation & Setup

### 1. Clone or Download the Repository

```bash
git clone https://github.com/your-username/ROCK-PAPER-SCISSOR-with-OpenCV-and-MediaPipe.git
cd ROCK-PAPER-SCISSOR-with-OpenCV-and-MediaPipe
```

Or simply download the project files into a directory.

### 2. Create a Virtual Environment (Recommended)

**Windows (PowerShell/CMD):**
```bash
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```


## 🕹️ Controls

### Start Screen

| Key | Action |
|---|---|
| `E` | Switch to **Easy** difficulty |
| `M` | Switch to **Medium** difficulty |
| `H` | Switch to **Hard** difficulty |
| `SPACE` | Start the game |
| `Q` | Quit |

### In-Game

| Key | Action |
|---|---|
| `Q` | Quit the game |

---

## 🤚 How to Play

1. Run the script and the **start screen** will appear.
2. Select your desired difficulty (`E`, `M`, or `H`) and press **SPACE** to begin.
3. Hold your hand in front of the webcam and show one of these gestures:

| Gesture | How to show it |
|---|---|
| ✊ **Rock** | Make a closed fist (0 fingers up) |
| ✋ **Paper** | Open all 5 fingers |
| ✌️ **Scissors** | Extend index + middle fingers only (2 fingers up) |

4. Hold the gesture **steady** for a moment — the game detects it automatically once it's stable.
5. The AI responds instantly and the result is shown on-screen.
6. Keep playing to rack up your score!

> **Tip:** Make sure only **one hand** is visible in the frame. Multiple hands will trigger an error.

---

## 🧠 How the AI Works

| Difficulty | Strategy |
|---|---|
| **Easy** | Picks a completely random move each round |
| **Medium** | Tracks your move history and counters your most-played gesture |
| **Hard** | Builds a **Markov chain** from your move sequences to predict — and beat — your next move |

---


## 🔍 Technical Deep Dive

### Hand Landmark Detection

MediaPipe Hands detects **21 landmarks** per hand:
- **0** — Wrist (center)
- **1-4** — Thumb (base to tip)
- **5-8** — Index finger (base to tip)
- **9-12** — Middle finger (base to tip)
- **13-16** — Ring finger (base to tip)
- **17-20** — Pinky finger (base to tip)

The model uses **floating-point coordinates** normalized to [0, 1] for each video frame.

### Gesture Classification Algorithm

The gesture recognition uses a **strict finger-counting approach**:

1. **Count raised fingers** using the `count_raised_fingers()` function:
   - Thumb: extended if its tip (landmark 4) is beyond the IP joint (landmark 3) horizontally
   - Index/Middle/Ring/Pinky: extended if tip is above PIP joint (smaller Y value in image space)

2. **Map finger count to move**:
   - `0 fingers` → Rock
   - `2 fingers` → Scissors
   - `5 fingers` → Paper
   - Any other count → Invalid (rejected)

3. **Validate with handedness**:
   - MediaPipe classifies each hand as "Left" or "Right"
   - Left-hand thumb logic is inverted (X-axis flipped)

### Markov Chain Prediction (Hard Mode)

The Markov chain model in Hard mode works as follows:

```python
markov[prev_move][next_move] = count
```

For example:
- If you played Rock → Paper → Rock → Paper → Scissors
- The transitions are tracked:
  - Rock → Paper (count = 2)
  - Paper → Rock (count = 1)
  - Rock → Scissors (count = 1)

When predicting your next move after "Paper":
- `markov["Paper"]` contains your likely next moves
- The AI picks the **most frequent outcome** and counters it

---

## 🎯 Game Flow Diagram

```
START
  ↓
[Start Screen] ← Select Difficulty (E/M/H)
  ↓
  SPACE → Enter Game Loop
  ↓
[Video Feed with HUD]
  ↓
Hand Detected? → No → Wait
  ↓ Yes
Valid Gesture? → No → Show error, wait for valid gesture
  ↓ Yes
Gesture Stable (8 frames)? → No → Increment stable counter
  ↓ Yes
Round Cooldown Elapsed? → No → Lock gesture, wait
  ↓ Yes
Play Round:
  ├─ Increment move history
  ├─ Update Markov chain (Hard mode)
  ├─ AI selects move based on difficulty
  ├─ Calculate winner
  └─ Update scores
  ↓
[Display Result on HUD]
  ↓
Q pressed? → Yes → EXIT
  ↓ No
Loop back to video feed
```

---
