"""
delay_tool.py — כלי יצירת קובץ שמע מדולי (Delay Tool)
=======================================================
מטרה:
    מדמה את ההפרש בזמן הגעה (TDOA) של גל קול בין שני מיקרופונים.
    מיקרופון חיצוני (reference) קולט את הרעש מוקדם יותר —
    לאחר שגל הקול עובר מרחק נוסף הוא מגיע למיקרופון הפנימי (primary).

    המרחק הטיפוסי בין מיקרופון חיצוני לפנימי באוזניה: ~1.5 ס"מ
    מהירות קול באוויר: ~343 מ'/שנ'
    זמן עיכוב:  1.5 ס"מ / 343 מ'/שנ' ≈ 43.7 מיקרו-שנ' ≈ 2 דגימות ב-44100Hz

שימוש:
    python delay_tool.py <input_file> [--delay 2] [--sr 44100] [--output delayed.wav]

פרמטרים:
    input_file  : נתיב לקובץ WAV המקורי
    --delay N   : מספר דגימות לדיליי (ברירת מחדל: 2)
    --sr HZ     : קצב דגימה (ברירת מחדל: 44100 — רק להמרה, הקובץ שומר על ה-sr המקורי)
    --output    : שם קובץ הפלט (ברירת מחדל: <input>_delayed.wav)
"""

import argparse
import os
import struct
import wave
import numpy as np


# --- קריאת קובץ WAV ---
# scipy.io.wavfile קורא נכון את כל סוגי WAV (int16/int32/float32)
def read_wav(path: str):
    """
    קורא קובץ WAV ומחזיר (sample_rate, audio_array_float64).
    audio_array ממוצע ל-mono אם סטריאו.
    """
    try:
        # נסיון ראשון: soundfile (תומך float, int24 וכד')
        import soundfile as sf
        data, sr = sf.read(path, dtype='float64')
    except ImportError:
        # fallback ל-scipy
        import scipy.io.wavfile as wavfile
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            # נרמול int16 → float64 בין -1 ל-1
            data = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float64) / 2147483648.0
        else:
            data = data.astype(np.float64)

    # המרה ל-mono: ממוצע כל הערוצים
    if data.ndim > 1:
        data = data.mean(axis=1)

    return sr, data


# --- כתיבת קובץ WAV ---
# wave module סטנדרטי — ללא תלויות חיצוניות
def write_wav(path: str, sr: int, data: np.ndarray):
    """
    כותב מערך float64 כקובץ WAV (int16, 16-bit PCM).
    מקצץ ערכים מחוץ לתחום [-1, 1] לפני המרה.
    """
    # קיצוץ ללא עיוות: ערכים מעל 1 → 1, מתחת ל-1- → -1-
    clipped = np.clip(data, -1.0, 1.0)

    # המרה ל-int16 (PCM 16-bit) — הפורמט הנפוץ ביותר
    pcm = (clipped * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wf:
        # 1 ערוץ (mono), 2 בתים לדגימה (16-bit), קצב הדגימה
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# --- פונקציית הדיליי המרכזית ---
def apply_delay(audio: np.ndarray, delay_samples: int) -> np.ndarray:
    """
    מוסיף דיליי של delay_samples דגימות לתחילת האות.
    הדגימות המאוחרות יותר "נחתכות" בסוף — האורך נשמר.

    לדוגמה, עם delay=2:
      קלט:  [a, b, c, d, e, f]
      פלט:  [0, 0, a, b, c, d]   (אורך זהה, 2 אפסים בהתחלה)

    זה מדמה שגל הקול הגיע 2 דגימות מאוחר יותר למיקרופון הפנימי.
    """
    if delay_samples <= 0:
        # אין דיליי — מחזיר עותק כדי לא לשנות את המקור
        return audio.copy()

    # יצירת מערך אפסים באורך הדגימות הדחויות
    silence = np.zeros(delay_samples, dtype=audio.dtype)

    # חיבור: אפסים + האות המקורי, וחיתוך לאורך המקורי
    delayed = np.concatenate([silence, audio])[:len(audio)]
    return delayed


# --- ממשק שורת הפקודה ---
def main():
    parser = argparse.ArgumentParser(
        description='יצירת קובץ WAV עם דיליי — מדמה TDOA בין מיקרופון חיצוני לפנימי'
    )
    parser.add_argument('input_file', help='נתיב לקובץ WAV המקורי')
    parser.add_argument(
        '--delay', type=int, default=2,
        help='מספר דגימות לדיליי (ברירת מחדל: 2 ≈ 1.5 ס"מ ב-44100Hz)'
    )
    parser.add_argument(
        '--sr', type=int, default=44100,
        help='קצב דגימה לתיעוד (הקובץ שומר על ה-sr המקורי, ברירת מחדל: 44100)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='נתיב קובץ הפלט (ברירת מחדל: <input>_delayed.wav)'
    )
    args = parser.parse_args()

    # בדיקת קיום קובץ קלט
    if not os.path.exists(args.input_file):
        print(f"[שגיאה] הקובץ לא נמצא: {args.input_file}")
        return 1

    # שם קובץ פלט ברירת מחדל
    if args.output is None:
        base, ext = os.path.splitext(args.input_file)
        args.output = base + '_delayed.wav'

    # --- קריאה ---
    print(f"[1/3] קורא קובץ: {args.input_file}")
    sr, audio = read_wav(args.input_file)
    duration_ms = len(audio) / sr * 1000
    print(f"      קצב דגימה: {sr} Hz | אורך: {duration_ms:.1f} ms | דגימות: {len(audio)}")

    # חישוב מרחק פיזי מקביל לדיליי
    delay_meters = args.delay / sr * 343.0
    delay_cm = delay_meters * 100
    delay_us = args.delay / sr * 1e6
    print(f"\n[2/3] מחיל דיליי:")
    print(f"      דגימות:  {args.delay}")
    print(f"      זמן:     {delay_us:.1f} מיקרו-שנ'")
    print(f"      מרחק:    {delay_cm:.2f} ס\"מ (מהירות קול 343 מ'/שנ')")

    # החלת הדיליי
    delayed_audio = apply_delay(audio, args.delay)

    # --- כתיבה ---
    print(f"\n[3/3] כותב קובץ: {args.output}")
    write_wav(args.output, sr, delayed_audio)
    print(f"      הצלחה! הקובץ נשמר: {args.output}")
    print(f"\n--- סיכום ---")
    print(f"  קובץ מקורי  (Primary / מיקרופון פנימי):  {args.input_file}")
    print(f"  קובץ מדולי  (Reference / מיקרופון חיצוני): {args.output}")
    print(f"  דיליי: {args.delay} דגימות = {delay_cm:.2f} ס\"מ")
    return 0


if __name__ == '__main__':
    exit(main())
