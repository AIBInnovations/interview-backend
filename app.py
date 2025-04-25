import os, threading, time, logging, cv2
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# detectors
from detectors.face   import analyze_face
from detectors.object import analyze_objects
from detectors.head   import analyze_head

# Gemini tasks
from interview.tasks import generate_question, evaluate_interview

app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO)


@app.route('/log_event', methods=['POST'])
def log_event():
    global warnings_count, last_warning_reason
    data   = request.get_json()
    event  = data.get('event')
    # map browser events to reasons
    if event == 'visibility_hidden':
        reason = 'Exam Window hidden'
    else:
        return jsonify(success=False), 400

    with warnings_lock:
        warnings_count += 1
        last_warning_reason = reason
        logging.info(f"Browser event warning #{warnings_count}: {reason}")
        if warnings_count >= 5:
            stop_event.set()
    return jsonify(success=True)


# ─── proctor state ───────────────────────────────────────────────────────────
warnings_count=0
last_warning_reason=None
warnings_lock=threading.Lock()
stop_event=threading.Event()
camera_thread=None

def proctor_loop():
    global warnings_count, last_warning_reason
    cap = cv2.VideoCapture(0)
    last_time = time.time()
    INTERVAL = 5.0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1); continue

        f = analyze_face(frame)
        o = analyze_objects(frame)
        h = analyze_head(frame)

        reason = None
        if f["flag"] != 1:
            reason = "Face error"
        elif any(d.get("warning") for d in o):
            reason = "Object detected"
        elif h.get("alert") and h.get("gaze") in ("Left","Right"):
            reason = f"Gaze {h['gaze']}"

        if reason and time.time() - last_time >= INTERVAL:
            with warnings_lock:
                warnings_count += 1
                last_warning_reason = reason
                last_time = time.time()
                logging.info(f"Warning #{warnings_count}: {reason}")
                if warnings_count >= 5:
                    stop_event.set()
        time.sleep(0.05)

    cap.release()
    logging.info("Proctor stopped")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        f = analyze_face(frame)
        for box in f.get("boxes", []):
            x1,y1,x2,y2 = map(int,box)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    cap.release()

# ─── React SPA ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# ─── Interview API ───────────────────────────────────────────────────────────
@app.route('/api/start', methods=['POST'])
def api_start():
    payload = request.get_json()
    job = payload.get('job_title','Candidate')
    q0 = generate_question(job, [], 0)
    return jsonify(question=q0)

@app.route('/api/question', methods=['POST'])
def api_question():
    data = request.get_json()
    q = generate_question(data['job_title'], data['history'], data['qnum'])
    return jsonify(question=q)

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    data = request.get_json()
    score = evaluate_interview(data['job_title'], data['history'])
    return jsonify(assessment=score)

# ─── Proctor API ─────────────────────────────────────────────────────────────
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_proctor', methods=['POST'])
def start_proctor():
    global camera_thread, warnings_count, last_warning_reason
    stop_event.clear()
    with warnings_lock:
        warnings_count=0
        last_warning_reason=None
    if not camera_thread or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=proctor_loop, daemon=True)
        camera_thread.start()
    return jsonify(sessionId=time.time())

@app.route('/status_proctor')
def status_proctor():
    return jsonify(
      warnings=warnings_count,
      reason=last_warning_reason,
      stopped=stop_event.is_set()
    )

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
