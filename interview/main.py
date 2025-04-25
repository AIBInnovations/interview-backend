# main.py

from interview_simulation import InteractiveInterview

def main():
    job_title = "UI Designer Basic Fresher"  # customize as needed
    sim = InteractiveInterview(job_title)
    sim.conduct_interview(num_questions=3)

if __name__ == "__main__":
    main()
