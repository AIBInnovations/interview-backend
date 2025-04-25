# interview_simulation.py

import json
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pyttsx3
import speech_recognition as sr

from tasks import generate_question, evaluate_interview

class InteractiveInterview:
    def __init__(self, job_title: str):
        self.job_title = job_title
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 3.0  # keep this
        self.microphone = sr.Microphone()
        # Executor for pipelining Gemini calls
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _listen_and_transcribe(self, phrase_time_limit: int = 10) -> str:
        with self.microphone as src:
            # shorten ambient noise calibration
            self.recognizer.adjust_for_ambient_noise(src, duration=0.3)
            try:
                audio = self.recognizer.listen(src, phrase_time_limit=phrase_time_limit)
                return self.recognizer.recognize_google(audio).strip().lower()
            except sr.UnknownValueError:
                return "[Could not understand speech]"
            except sr.RequestError as e:
                return f"[STT request failed: {e}]"

    def conduct_interview(self, num_questions: int = 5):
        interview_history = []

        # --- Intro Q1 ---
        intro = "Let's start with a brief introduction. Can you tell me about yourself?"
        print(f"\nAI: {intro}")
        self.engine.say(intro); self.engine.runAndWait()
        print("Listening for your introduction…")
        answer = self._listen_and_transcribe()

        if answer in {"repeat", "can you repeat", "say it again"}:
            print("Replaying intro…")
            self.engine.say(intro); self.engine.runAndWait()
            answer = self._listen_and_transcribe()

        print(f"You: {answer}")
        interview_history.append({"question": intro, "answer": answer})

        # Pre-schedule Q2 while user answers (batch/pipeline)
        future_q = self.executor.submit(
            generate_question,
            self.job_title,
            interview_history,
            2
        )

        # --- Role-specific Q2–Q(N+1) ---
        for i in range(num_questions):
            qnum = i + 2

            # 1) Wait for or generate current question
            question = future_q.result()
            print(f"\nAI: {question}")
            self.engine.say(question); self.engine.runAndWait()

            # 2) Immediately kick off next question generation
            future_q = self.executor.submit(
                generate_question,
                self.job_title,
                interview_history + [{"question": question, "answer": None}],
                qnum + 1
            )

            # 3) Listen with phrase_time_limit
            response = self._listen_and_transcribe()
            # handle repeats
            m = re.match(r"(?:repeat|say it again) question (\d+)", response)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(interview_history):
                    prev_q = interview_history[idx]["question"]
                    print(f"Replaying Q{idx+1}: {prev_q}")
                    self.engine.say(prev_q); self.engine.runAndWait()
                    response = self._listen_and_transcribe()
                    interview_history[idx]["answer"] = response
                    print(f"You (revised): {response}")
                    continue

            if response in {"repeat", "say it again"} and interview_history:
                prev_q = interview_history[-1]["question"]
                print("Replaying previous question…")
                self.engine.say(prev_q); self.engine.runAndWait()
                response = self._listen_and_transcribe()
                interview_history[-1]["answer"] = response
                print(f"You (revised): {response}")
                continue

            print(f"You: {response}")
            interview_history.append({"question": question, "answer": response})

        # --- Final evaluation ---
        print("\nAI is evaluating your responses…\n")
        assessment = evaluate_interview(self.job_title, interview_history)
        print("=== Assessment ===\n")
        print(assessment)

        # Save results
        results = {
            "job_title": self.job_title,
            "interview_date": datetime.now().isoformat(),
            "interview_history": interview_history,
            "assessment": assessment
        }
        fname = f"interactive_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {fname}")

        return assessment
