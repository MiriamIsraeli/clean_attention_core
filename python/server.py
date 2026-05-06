"""
Clean Attention — שרת זמן אמת (ANC דיגיטלי)
=============================================

ארכיטקטורה:
  המשתמש מעלה תיקייה עם 2 קבצי WAV:
    primary.wav   — ערוץ ראשי   (= מיקרופון פנימי: רעש + דיבור)
    reference.wav — ערוץ ייחוס  (= מיקרופון חיצוני: אותו רעש + דיליי)

  תהליך:
    1. NLMSFilter לומד את הרעש מה-reference
    2. y[n] = אמידת הרעש (גל ביטול = -y[n])
    3. e[n] = primary[n] - y[n]  התכנס לדיבור בלבד
    4. VAD: בזמן דיבור משקולות קפואות
    5. הדפדפן מקבל: antiNoise + error + original
"""
import os, io, json, time, uuid, threading
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response
from algorithms import NLMSFilter, LMSFilter, FxLMSFilter

try:
    import imageio_ffmpeg
    os.environ['PATH'] = (os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) +
                          os.pathsep + os.environ.get('PATH', ''))
except ImportError:
    pass

app = Flask(__name__)
WEB_DIR = os.path.join(os.path.dirname(__file__), '..', 'web')


def read_wav_file(path):
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float64')
        if data.ndim > 1:
            data = data.mean(axis=1)
        return sr, data
    except Exception:
        pass
    import scipy.io.wavfile as wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data


def read_audio_fileobj(file_obj):
    buf = io.BytesIO(file_obj.read())
    filename = getattr(file_obj, 'filename', '') or ''
    try:
        import scipy.io.wavfile as wavfile
        buf.seek(0)
        sr, audio = wavfile.read(buf)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float64) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float64) / 2147483648.0
        else:
            audio = audio.astype(np.float64)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return sr, audio
    except Exception:
        pass
    try:
        import soundfile as sf
        buf.seek(0)
        audio, sr = sf.read(buf)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return sr, audio.astype(np.float64)
    except Exception:
        pass
    try:
        import subprocess, tempfile
        ffmpeg_path = 'ffmpeg'
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        buf.seek(0)
        ext = os.path.splitext(filename)[1] or '.mp3'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [ffmpeg_path, '-i', tmp_path, '-f', 's16le',
                 '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', '-'],
                capture_output=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors='ignore'))
            audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float64) / 32768.0
            return 44100, audio
        finally:
            os.unlink(tmp_path)
    except Exception:
        pass
    raise ValueError('פורמט לא נתמך')


def open_wav_stream(file_obj):
    """
    פותח קובץ WAV כ-stream של float64.
    מחזיר: (stream, sr, total_samples)

    הקובץ נטען פעם אחת (מגבלת HTTP), אך ה-stream מאפשר
    קריאה chunk אחר chunk בלבד — בדיוק כמו מיקרופון אמיתי.
    """
    buf = io.BytesIO(file_obj.read())
    buf.seek(0)

    # נסיון עם scipy
    try:
        import scipy.io.wavfile as wavfile
        buf.seek(0)
        sr, data = wavfile.read(buf)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float64) / 2147483648.0
        else:
            data = data.astype(np.float64)
        return io.BytesIO(data.tobytes()), sr, len(data)
    except Exception:
        pass

    # נסיון עם soundfile
    try:
        import soundfile as sf
        buf.seek(0)
        data, sr = sf.read(buf, dtype='float64')
        if data.ndim > 1:
            data = data.mean(axis=1)
        return io.BytesIO(data.tobytes()), sr, len(data)
    except Exception:
        pass

    raise ValueError('פורמט לא נתמך')


def read_chunk_from_stream(stream, n_samples):
    """
    קורא בדיוק n_samples דגימות (float64) מה-stream.
    מחזיר מערך numpy, או None אם הגענו לסוף.
    המידע לא יודע מה נמצא אחרי ה-chunk הנוכחי.
    """
    raw = stream.read(n_samples * 8)  # 8 bytes לכל float64
    if not raw:
        return None
    chunk = np.frombuffer(raw, dtype=np.float64)
    return chunk if len(chunk) > 0 else None


@app.route('/')
def index():
    return send_from_directory(WEB_DIR, 'index.html')


@app.route('/anc-stream', methods=['POST'])
def anc_stream():
    """
    ביטול רעשים אקטיבי בזמן אמת (NLMS + VAD).

    קלט (multipart/form-data):
      primary   : WAV ערוץ ראשי (מיקרופון פנימי)
      reference : WAV ערוץ ייחוס (מיקרופון חיצוני + דיליי)
      filter_len: אורך מסנן NLMS (ברירת מחדל 256)
      mu        : קצב למידה (ברירת מחדל 0.01)
      vad_thresh: סף VAD (ברירת מחדל 0.02)

    פלט: Server-Sent Events עם chunks מעובדים
    """
    if 'primary' not in request.files or 'reference' not in request.files:
        return jsonify({'error': 'נדרשים 2 קבצים: primary ו-reference'}), 400

    filter_len = int(request.form.get('filter_len', 256))
    mu         = float(request.form.get('mu', 0.01))
    vad_ratio  = float(request.form.get('vad_thresh', 1.5))

    try:
        prim_stream, sr_p, prim_total = open_wav_stream(request.files['primary'])
        ref_stream,  sr_r, ref_total  = open_wav_stream(request.files['reference'])
    except Exception as ex:
        return jsonify({'error': str(ex)}), 400

    sr = sr_p
    chunk_samples     = max(2048, int(sr * 0.05) // 256 * 256)
    # כמה דגימות reference מקבילות ל-chunk_samples של primary (לפי יחס קצב הדגימה)
    ref_chunk_samples = max(1, int(chunk_samples * sr_r / sr_p))
    total_samples     = min(prim_total, int(ref_total * sr_p / sr_r))

    def generate():
        nlms = NLMSFilter(N=filter_len, mu=mu, sr=sr, vad_ratio=vad_ratio)

        yield f"data: {json.dumps({'type': 'meta', 'sampleRate': sr, 'totalSamples': total_samples, 'chunkSize': chunk_samples, 'filterLen': filter_len, 'mu': mu, 'vadThresh': vad_ratio})}\n\n"

        chunk_id = 0
        t_start = time.time()
        processed_samples = 0
        n_noise_chunks = 0
        total_noise_pri_power = 0.0
        total_noise_err_power = 0.0

        while True:
            # === קריאת chunk — המידע לא יודע מה נמצא אחריו ===
            # בדיוק כמו מיקרופון אמיתי שמספק רק את הרגע הנוכחי
            prim_chunk = read_chunk_from_stream(prim_stream, chunk_samples)
            ref_raw    = read_chunk_from_stream(ref_stream,  ref_chunk_samples)

            if prim_chunk is None or ref_raw is None:
                break

            # סנכרון קצבי דגימה — resample רק את ה-chunk הנוכחי
            if sr_r != sr_p:
                ref_chunk = np.interp(
                    np.linspace(0, len(ref_raw) - 1, len(prim_chunk)),
                    np.arange(len(ref_raw)),
                    ref_raw
                )
            else:
                ref_chunk = ref_raw

            # וודא אורכים שווים (chunk אחרון עשוי להיות קצר יותר)
            min_len    = min(len(prim_chunk), len(ref_chunk))
            prim_chunk = prim_chunk[:min_len]
            ref_chunk  = ref_chunk[:min_len]

            # === ליבת ה-ANC: NLMS + VAD ===
            # ref_chunk  = x[n] — reference (מיקרופון חיצוני)
            # prim_chunk = d[n] — primary   (מיקרופון פנימי)
            # y[n]   = w^T·x[n]              אמידת הרעש
            # e[n]   = d[n] - y[n]            שגיאה = דיבור בלבד
            # anti   = -y[n]                  גל ביטול לאוזניות
            noise_est, error, anti_noise = nlms.process_chunk(ref_chunk, prim_chunk)

            step = max(1, len(prim_chunk) // 200)
            rms_p = float(np.sqrt(np.mean(prim_chunk  ** 2) + 1e-12))
            rms_a = float(np.sqrt(np.mean(anti_noise  ** 2) + 1e-12))
            rms_e = float(np.sqrt(np.mean(error       ** 2) + 1e-12))

            processed_samples += len(prim_chunk)

            # צובר רק chunks של רעש טהור (ללא דיבור) למדד dB
            if not nlms.voice_active:
                total_noise_pri_power += float(np.mean(prim_chunk ** 2))
                total_noise_err_power += float(np.mean(error      ** 2))
                n_noise_chunks += 1

            payload = json.dumps({
                'type':        'chunk',
                'id':          chunk_id,
                'original':    prim_chunk[::step].tolist(),
                'antiNoise':   anti_noise[::step].tolist(),
                'error':       error[::step].tolist(),
                'errorAudio':  error.tolist(),
                'rmsPrimary':  rms_p,
                'rmsAnti':     rms_a,
                'rmsError':    rms_e,
                'stepFactor':  step,
                'voiceActive': nlms.voice_active
            })
            yield f"data: {payload}\n\n"

            chunk_id += 1

            if chunk_id > 5:
                target = t_start + (processed_samples / sr) * 0.45
                wait = target - time.time()
                if wait > 0:
                    time.sleep(wait)

        noise_red_db = 0.0
        if n_noise_chunks > 0 and total_noise_pri_power > 1e-12:
            noise_red_db = -10.0 * np.log10(
                total_noise_err_power / total_noise_pri_power + 1e-12)

        yield f"data: {json.dumps({'type': 'done', 'totalChunks': chunk_id, 'noiseReductionDb': round(float(noise_red_db), 1), 'processingTimeSec': round(time.time() - t_start, 2)})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/delay', methods=['POST'])
def apply_delay_endpoint():
    """
    מקבל קובץ WAV ומחזיר primary_delayed.wav:
      - דיליי קטן (מדמה הגעת הצליל מאוחר יותר למיקרופון הפנימי)
      - הנחתה של 15% (מדמה בליעת הצליל על ידי כוס האוזנייה)
    קלט: audio (WAV), delay_samples (int, ברירת מחדל 2)
    פלט: WAV מדולי + מוחלש = primary_delayed.wav
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'חסר קובץ audio'}), 400

    delay_samples = int(request.form.get('delay_samples', 2))
    attenuation   = float(request.form.get('attenuation', 0.85))  # 85% = הנחתה של 15%

    try:
        sr, audio = read_audio_fileobj(request.files['audio'])
    except Exception as ex:
        return jsonify({'error': str(ex)}), 400

    # דיליי: הקובץ הפנימי מגיע מאוחר יותר (אחרי שעבר דרך כוס האוזנייה)
    silence = np.zeros(delay_samples, dtype=np.float64)
    delayed = np.concatenate([silence, audio])[:len(audio)]

    # הנחתה: כוס האוזנייה מחלישה את הצליל החודר פנימה
    delayed = delayed * attenuation

    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (np.clip(delayed, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)

    delay_cm = delay_samples / sr * 343.0 * 100
    resp = Response(buf.read(), mimetype='audio/wav')
    resp.headers['Content-Disposition'] = 'attachment; filename=primary_delayed.wav'
    resp.headers['X-Delay-Samples'] = str(delay_samples)
    resp.headers['X-Delay-CM'] = f'{delay_cm:.2f}'
    return resp


if __name__ == '__main__':
    print("=" * 55)
    print("  Clean Attention - ANC Real-Time Simulation")
    print("  פתח דפדפן: http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000, threaded=True)
