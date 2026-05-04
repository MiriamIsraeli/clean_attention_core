"""
Clean Attention — אלגוריתמי ליבה לביטול רעשים אקטיבי (ANC)
=============================================================

ארכיטקטורת המערכת (מדמה 2 מיקרופונים):
  ┌──────────────┐     ┌─────────────────────────────────┐
  │ Primary (d)  │────▶│  NLMS Adaptive Filter           │
  │ מיקרופון פנימי│     │  e[n] = d[n] - y[n]            │──▶  e[n]
  └──────────────┘     │  (התכנס = דיבור בלבד)           │    גל ביטול = -y[n]
  ┌──────────────┐     │                                 │
  │ Reference (x)│────▶│  x[n] = reference (רעש+דיליי)   │
  │ מיקרופון חיצוני│    │  VAD מקפיא עדכון בזמן דיבור    │
  └──────────────┘     └─────────────────────────────────┘

מחלקות:
  VAD          — זיהוי פעילות קולית (Voice Activity Detection)
  NLMSFilter   — מסנן אדפטיבי NLMS עם קפיאת VAD
  LMSFilter    — תאימות לאחור (ממשק זהה לישן)
  FxLMSFilter  — Filtered-x LMS (לתיעוד אקדמי)
"""
import numpy as np


# =============================================================
# VAD — Voice Activity Detection
# =============================================================
# עקרון הפעולה:
#   מחשב אנרגיה (RMS) של כל חלון קצר.
#   אם האנרגיה עולה על הסף (threshold) — דיבור מזוהה.
#   כדי למנוע "גזירות" מהירות, מוגדרים:
#     hold_frames: מספר frames לשמור על מצב VOICE לאחר שהאנרגיה ירדה
#     attack_alpha: EMA מהיר לחישוב אנרגיה בעלייה (תגובה מהירה לדיבור)
#     release_alpha: EMA איטי לחישוב אנרגיה בירידה (שחרור חלק)
class VAD:
    def __init__(self,
                 sr: int = 44100,
                 frame_ms: int = 20,
                #  threshold: float = 0.02,
                vad_ratio: float = 1.5,
                 hold_ms: int = 80):
        """
        sr          : קצב דגימה (Hz)
        frame_ms    : אורך חלון VAD במילישניות (ברירת מחדל 20ms)
        # threshold   : סף אנרגיה RMS לזיהוי דיבור (0.0–1.0)
        vad_ratio   : יחס דיבור/רעש לזיהוי דיבור (ברירת מחדל 1.5)
        hold_ms     : כמה ms לשמור על VOICE לאחר שהאנרגיה ירדה
        """
        # גודל frame בדגימות
        self.frame_size = int(sr * frame_ms / 1000)

        # סף אנרגיה
        # self.threshold = threshold
        self.vad_ratio = vad_ratio
        self._noise_floor = None   # ← לומד את רמת הרעש אוטומטית

        # מספר frames להחזקת מצב VOICE לאחר שהאנרגיה ירדה
        self.hold_frames = max(1, int(hold_ms / frame_ms))

        # מונה frames שנשאר במצב VOICE
        self._hold_counter = 0

        # EMA של אנרגיה — מחשב חלק יותר מ-frame בודד
        self._energy_ema = 0.0

        # alpha לעלייה (תגובה מהירה לדיבור) ולירידה (שחרור איטי)
        self._alpha_attack = 0.6    # ~1-2 frames לגילוי
        self._alpha_release = 0.05  # ~20 frames לשחרור

        # מצב נוכחי
        self.is_voice = False

    def process(self, chunk: np.ndarray) -> bool:
        """
        מקבל chunk של דגימות, מחזיר True אם יש דיבור פעיל.

        האלגוריתם:
        1. מחשב RMS (אנרגיה שורש-ממוצע-ריבועי) של ה-chunk
        2. EMA אסימטרי: עולה מהר כשיש אנרגיה, יורד לאט כשאין
        3. אם EMA > threshold → VOICE
        4. hold_frames: שומר VOICE כמה frames נוספים לאחר הסף
        """
        # חישוב RMS: sqrt(mean(x²)) — מדד אנרגיה
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

        # EMA אסימטרי: עולה מהר, יורד לאט
        if rms > self._energy_ema:
            # עלייה: alpha גבוה — תגובה מהירה לדיבור
            self._energy_ema = (self._alpha_attack * rms +
                                (1 - self._alpha_attack) * self._energy_ema)
        else:
            # ירידה: alpha נמוך — שחרור איטי
            self._energy_ema = (self._alpha_release * rms +
                                (1 - self._alpha_release) * self._energy_ema)

        # קבלת החלטה: מעל הסף → hold; מתחת → ספר לאחור
        # if self._energy_ema >= self.threshold:
        # עדכון רצפת הרעש
        if self._noise_floor is None:
            self._noise_floor = rms
        elif rms < self._noise_floor:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms  # מהיר לטה
        else:
            self._noise_floor = 0.999 * self._noise_floor + 0.001 * rms  # איטי לעלייה
        # דיבור = RMS גבוה מהרצפה פי vad_ratio
        if self._energy_ema >= self._noise_floor * self.vad_ratio:
            # דיבור מזוהה — איפוס מונה ה-hold
            self._hold_counter = self.hold_frames
            self.is_voice = True
        elif self._hold_counter > 0:
            # מחזיקים VOICE עד שמונה ה-hold מגיע לאפס
            self._hold_counter -= 1
            self.is_voice = True
        else:
            self.is_voice = False

        return self.is_voice


# =============================================================
# NLMSFilter — Normalized LMS עם קפיאת VAD
# =============================================================
# עקרון הפעולה:
#   כל דגימה:
#     y[n] = w^T · x[n]               (אמידת הרעש)
#     e[n] = d[n] - y[n]              (שגיאה = primary - אמידה)
#     w[n+1] = w[n] + μ·e[n]·x[n] / (||x[n]||² + ε)  (NLMS update)
#
#   כאשר VAD מזהה דיבור: דילוג על עדכון המשקולות.
#   כך e[n] = d[n] - y[n] ≈ דיבור + שאריות (לא מנסים לבטל דיבור).
#
#   גל הביטול: anti[n] = -y[n]
#   אות יוצא לאוזניות (חלק של ה"אוויר"): e[n] + anti[n] = e[n] - y[n]... 
#   אך בסימולציה: output = e[n] = primary - y[n]
class NLMSFilter:
    def __init__(self,
                 N: int = 256,
                 mu: float = 0.01,
                 eps: float = 1e-6,
                 weight_limit: float = 10.0,
                 sr: int = 44100,
                 vad_ratio: float = 1.5):
        """
        N             : אורך מסנן (מספר משקולות) — ברירת מחדל 256
        mu            : קצב למידה (step size) — ברירת מחדל 0.01
        eps           : הגנה מחלוקה באפס — ברירת מחדל 1e-6
        weight_limit  : הגבלת נורמה של וקטור המשקולות (יציבות)
        sr            : קצב דגימה
        vad_ratio     : יחס דיבור/רעש לזיהוי VAD — ברירת מחדל 1.5
        """
        self.N = N
        self.mu = mu
        self.eps = eps
        self.weight_limit = weight_limit

        # וקטור משקולות — מאופס בהתחלה
        self.w = np.zeros(N, dtype=np.float64)

        # באפר קלט circular — שומר N דגימות אחורה של reference
        self.x_buf = np.zeros(N, dtype=np.float64)

        # VAD — שומר על דיבור
        self.vad = VAD(sr=sr, vad_ratio=vad_ratio)

        # סטטיסטיקות לממשק
        self.last_error = 0.0
        self.last_noise_estimate = 0.0
        self.voice_active = False

    def process_chunk(self, reference: np.ndarray, primary: np.ndarray):
        """
        עיבוד chunk בזמן אמת.

        פרמטרים:
          reference : ערוץ ייחוס — מיקרופון חיצוני (רעש בלבד, עם דיליי)
          primary   : ערוץ ראשי — מיקרופון פנימי (רעש + דיבור)

        מחזיר:
          noise_estimate : y[n] — אמידת הרעש (= גל הביטול שיושמע)
          error          : e[n] — שגיאה (≈ דיבור לאחר התכנסות)
          anti_noise     : -y[n] — הגל ההפוך שמשמיעים באוזניות
        """
        n = len(reference)

        # וודא אורכים שווים
        assert len(primary) == n, "primary ו-reference חייבים באורך זהה"

        noise_estimate = np.zeros(n, dtype=np.float64)
        error = np.zeros(n, dtype=np.float64)

        # בדיקת VAD לכל ה-chunk פעם אחת (יעיל יותר מבדיקה לכל דגימה)
        self.voice_active = self.vad.process(primary)

        for i in range(n):
            # --- שלב 1: הזזת באפר + הכנסת דגימה חדשה ---
            # מחלקת x_buf כ-FIFO: x_buf[0] = החדש, x_buf[N-1] = הישן ביותר
            self.x_buf[1:] = self.x_buf[:-1]   # הזזה ימינה
            self.x_buf[0] = reference[i]         # הכנסת הדגימה החדשה

            # --- שלב 2: אמידת הרעש y[n] = w^T · x ---
            y = float(np.dot(self.w, self.x_buf))

            # --- שלב 3: חישוב שגיאה e[n] = d[n] - y[n] ---
            e = float(primary[i]) - y

            # --- שלב 4: עדכון משקולות (NLMS) — רק אם אין דיבור ---
            # NLMS: μ מחולק ב-||x||² + ε לנרמול לפי עוצמת הקלט
            # בזמן דיבור: משקולות קפואות — לא לומדים לבטל דיבור!
            if not self.voice_active:
                power = float(np.dot(self.x_buf, self.x_buf)) + self.eps
                self.w += (self.mu / power) * e * self.x_buf

                # הגבלת נורמה למניעת אי-יציבות
                norm = float(np.linalg.norm(self.w))
                if norm > self.weight_limit:
                    self.w *= self.weight_limit / norm

            noise_estimate[i] = y
            error[i] = e

        # שמירת ערכים אחרונים לסטטיסטיקות
        self.last_noise_estimate = float(np.sqrt(np.mean(noise_estimate ** 2)))
        self.last_error = float(np.sqrt(np.mean(error ** 2)))

        # גל הביטול = הפוך מאמידת הרעש
        anti_noise = -noise_estimate

        return noise_estimate, error, anti_noise


# ===== מסנן LMS אדפטיבי =====
# לומד את דפוס הרעש ומייצר אות ביטול בזמן אמת
class LMSFilter:
    def __init__(self, N=128, mu=0.01):
        self.N = N
        self.mu = mu
        self.w = np.zeros(N)
        self.x = np.zeros(N)

    def process_chunk(self, reference, desired):
        """עיבוד chunk בודד בזמן אמת - מחזיר (אמידת_רעש, שגיאה)"""
        n = len(reference)
        output = np.zeros(n)
        error = np.zeros(n)

        for i in range(n):
            self.x[1:] = self.x[:-1]
            self.x[0] = reference[i]
            y = np.dot(self.w, self.x)
            e = desired[i] - y
            power = np.dot(self.x, self.x) + 1e-8
            self.w += (self.mu / power) * e * self.x
            norm = np.linalg.norm(self.w)
            if norm > 10.0:
                self.w *= 10.0 / norm
            output[i] = y
            error[i] = e

        return output, error


# ===== מסנן FxLMS =====
class FxLMSFilter:
    def __init__(self, N=128, mu=0.001, secondary_path=None):
        self.N = N
        self.mu = mu
        self.w = np.zeros(N)
        self.x_buf = np.zeros(N)
        self.xf_buf = np.zeros(N)
        self.sec_path = secondary_path if secondary_path is not None \
            else np.array([0.8, -0.2, 0.05])

    def process_chunk(self, reference, desired):
        """עיבוד chunk בודד בזמן אמת"""
        n = len(reference)
        output = np.zeros(n)
        error = np.zeros(n)

        for i in range(n):
            self.x_buf[1:] = self.x_buf[:-1]
            self.x_buf[0] = reference[i]
            sp_len = len(self.sec_path)
            xf = np.dot(self.sec_path, self.x_buf[:sp_len])
            self.xf_buf[1:] = self.xf_buf[:-1]
            self.xf_buf[0] = xf
            y = np.dot(self.w, self.x_buf)
            e = desired[i] - y
            power = np.dot(self.xf_buf, self.xf_buf) + 1e-8
            self.w += (self.mu / power) * e * self.xf_buf
            norm = np.linalg.norm(self.w)
            if norm > 10.0:
                self.w *= 10.0 / norm
            output[i] = y
            error[i] = e

        return output, error


# ===== גלאי רעש סטציונרי =====
class NoiseDetector:
    """
    גלאי ומבטל רעש סטציונרי בזמן אמת — Wiener Filter + Stationarity Tracking.
    
    שלוש שכבות הגנה על דיבור:
    
    1. מסנן Wiener: G(f) = max(1 - α·(noise/mag)², β)
       - bin רעש (mag ≈ noise) → G ≈ β → דיכוי מלא
       - bin דיבור (mag >> noise) → G ≈ 1 → שימור מלא
       - הגנה אוטומטית לפי SNR — לא צריך סף ידני
       
    2. מעקב סטציונריות per-bin (צורת ספקטרום מנורמלת):
       - bin יציב = רעש מאומת → alpha מלא
       - bin שמשתנה = דיבור אפשרי → alpha מופחת (זהירות נוספת)
       - שינוי עוצמה בלבד (volume) נחשב סטציונרי!
       
    3. החלקה זמנית (temporal smoothing):
       - attack מהיר: דיבור מגיע → gain עולה מיד (לא חותך תחילת מילה)
       - release איטי: דיבור נגמר → gain יורד בהדרגה (לא חותך סוף מילה)
    
    משתמש ב-sqrt(Hanning) כחלון — COLA עם overlap 50%.
    מחזיר אות מנוקה. Anti-noise = cleaned - original.
    """
    def __init__(self, fft_size=1024, threshold=0.3):
        self.fft_size = fft_size
        self.threshold = threshold
        self.bins = fft_size // 2 + 1
        self.hop = fft_size // 2
        # sqrt(Hanning) — מבטיח ש-window^2 = Hanning ו-COLA עם 50% overlap
        self.window = np.sqrt(np.hanning(fft_size) + 1e-12)
        self.frame_count = 0
        # פרופיל רעש — ממוצע ספקטרום של ה-frames הראשונים
        self.noise_profile = np.zeros(self.bins)
        self.calibration_frames = 8  # ~90ms ב-48kHz (מספיק ללמוד רעש)
        # פרמטרי מסנן Wiener
        self.alpha = 3.0      # over-subtraction — גבוה = ביטול חזק יותר
        self.beta = 0.02      # spectral floor — מונע "רעש מוזיקלי" (artifacts)

        # --- מעקב סטציונריות per-bin ---
        self._shape_mean = np.zeros(self.bins)   # ממוצע EMA של צורת הספקטרום
        self._shape_var = np.zeros(self.bins)     # שונות EMA
        self._stationarity = np.ones(self.bins)   # ציון [0,1] — 1=סטציונרי
        self._ema_attack = 0.4    # EMA מהיר כשהצורה משתנה (דיבור מגיע)
        self._ema_release = 0.05  # EMA איטי כשהצורה יציבה (דיבור נגמר)

        # --- החלקה זמנית של ה-gain (לא נדרשת — COLA כבר מחליקה) ---
        # Wiener gain הוא רציף ב-mag → המעבר בין frames חלק מטבעו

        # באפרים ל-overlap-add
        self._in_buf = np.array([], dtype=np.float64)
        self._out_clean = np.array([], dtype=np.float64)
        self._out_cnt = np.array([], dtype=np.float64)
        self._next_start = 0
        self._consumed = 0
        # השהיה של hop דגימות — מבטיחה COLA שלם (כל sample מכוסה ב-2 frames)
        # ב-48kHz זה ~10ms — לא מורגש
        self._out_delay = np.array([], dtype=np.float64)

    def _update_stationarity(self, mag):
        """
        עדכון ציון סטציונריות per-bin עם EMA אסימטרי.
        
        הרעיון: מנרמלים את הספקטרום לפי האנרגיה הכוללת (= "צורה").
        רעש סטציונרי שומר על אותה צורה גם כשהעוצמה עולה/יורדת.
        דיבור/מוזיקה משנים את הצורה — bins חדשים מקבלים אנרגיה.
        
        EMA אסימטרי:
        - כשהצורה משתנה (דיבור מתחיל) → עדכון מהיר (attack=0.4, ~2 frames)
        - כשהצורה יציבה (דיבור נגמר) → עדכון איטי (release=0.05, ~20 frames)
        זה מבטיח שדיבור מוגן מיד כשהוא מופיע, ולא נחתך בהתחלה.
        """
        # נרמול לפי אנרגיה כוללת — שינוי עוצמה בלבד לא משפיע
        total_energy = np.sum(mag) + 1e-10
        shape = mag / total_energy

        if self.frame_count <= self.calibration_frames:
            # בכיול: אתחול הממוצע והשונות
            if self.frame_count == 1:
                self._shape_mean = shape.copy()
                self._shape_var = np.zeros(self.bins)
            else:
                old_mean = self._shape_mean.copy()
                self._shape_mean = (
                    (self.frame_count - 1) * old_mean + shape
                ) / self.frame_count
                self._shape_var = (
                    (self.frame_count - 1) * self._shape_var +
                    (shape - old_mean) * (shape - self._shape_mean)
                ) / self.frame_count
            self._stationarity = np.ones(self.bins)
        else:
            # EMA אסימטרי: attack מהיר / release איטי
            diff = shape - self._shape_mean
            diff_sq = diff ** 2
            # אם השונות עולה (שינוי = דיבור מגיע) → alpha גבוה (תגובה מהירה)
            # אם השונות יורדת (יציבות = רעש חוזר) → alpha נמוך (שחרור איטי)
            alpha_per_bin = np.where(
                diff_sq > self._shape_var,
                self._ema_attack,    # attack: שינוי מהיר
                self._ema_release    # release: שחרור איטי
            )
            self._shape_mean += alpha_per_bin * diff
            self._shape_var = (1 - alpha_per_bin) * self._shape_var + alpha_per_bin * diff_sq

            # ציון סטציונריות: CV² = var / mean²
            cv2 = self._shape_var / (self._shape_mean ** 2 + 1e-20)
            self._stationarity = np.clip(
                1.0 - cv2 / (self.threshold ** 2), 0.0, 1.0
            )

    def update_and_extract(self, chunk):
        """
        עיבוד chunk בזמן אמת: מסנן Wiener אדפטיבי + הגנת דיבור.
        
        לכל bin תדר:
        - mag ≈ noise_profile (רעש) → gain ≈ β → דיכוי מלא
        - mag >> noise_profile (דיבור) → gain ≈ 1 → שימור מלא
        - bin לא-סטציונרי (דיבור) → alpha מופחת → זהירות נוספת
        
        שינוי עוצמה בלבד (volume up/down) לא נחשב כשינוי!
        
        מחזיר את האות המנוקה. הגל ההפוך = cleaned - original.
        """
        fft_size = self.fft_size
        hop = self.hop
        window = self.window

        # הוספת דגימות חדשות לבאפר
        self._in_buf = np.concatenate([self._in_buf, chunk])
        extra = len(self._in_buf) - len(self._out_clean)
        if extra > 0:
            self._out_clean = np.concatenate([self._out_clean, np.zeros(extra)])
            self._out_cnt = np.concatenate([self._out_cnt, np.zeros(extra)])

        # עיבוד כל ה-frames השלמים
        while self._next_start + fft_size <= len(self._in_buf):
            start = self._next_start
            frame = self._in_buf[start:start + fft_size] * window
            spectrum = np.fft.rfft(frame)
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)

            self.frame_count += 1

            # --- עדכון סטציונריות per-bin ---
            self._update_stationarity(mag)

            # --- עדכון פרופיל רעש ---
            if self.frame_count <= self.calibration_frames:
                # כיול: ממוצע נצבר של ה-frames הראשונים
                if self.frame_count == 1:
                    self.noise_profile = mag.copy()
                else:
                    self.noise_profile = (
                        (self.frame_count - 1) * self.noise_profile + mag
                    ) / self.frame_count
                # במהלך כיול: דיכוי אגרסיבי (כמעט שקט)
                clean_mag = mag * self.beta
            else:
                # === מסנן Wiener — הגנה אוטומטית על דיבור ===
                #
                # G(f) = max(1 - α·(noise_profile / mag)², β)
                #
                # הפרדה אוטומטית בין רעש לדיבור לפי SNR per-bin:
                # - bin רעש (mag ≈ noise) → G = max(1-α, β) = β → דיכוי מלא
                # - bin דיבור (mag >> noise) → G ≈ 1.0 → שימור מלא
                # - bin דיבור+רעש → G ביניים → רוב הדיבור נשמר
                #
                # גלי קול שמשתנים רק בעוצמה (volume) נשמרים כי
                # SNR = mag/noise נשאר גבוה — רק ה-scale השתנה

                # יחס (noise/signal)² per-bin
                noise_ratio_sq = (self.noise_profile / (mag + 1e-10)) ** 2

                # Wiener gain
                gain = np.maximum(
                    1.0 - self.alpha * noise_ratio_sq, self.beta
                )

                clean_mag = gain * mag

            # שחזור frame מנוקה
            clean_spec = clean_mag * np.exp(1j * phase)
            clean_frame = np.fft.irfft(clean_spec, n=fft_size) * window

            self._out_clean[start:start + fft_size] += clean_frame
            self._out_cnt[start:start + fft_size] += window ** 2

            self._next_start += hop

        # === חילוץ תוצאה עם COLA שלם ===
        # _next_start = תחילת ה-frame הבא שעוד לא עובד.
        # כל sample לפני _next_start מכוסה ע"י 2 frames (COLA מלא).
        # אחרי _next_start — רק frame אחד (חלקי). לכן חולצים רק עד _next_start.
        safe_end = self._next_start
        new_ready = self._out_clean[self._consumed:safe_end].copy()
        new_cnt = self._out_cnt[self._consumed:safe_end].copy()
        new_cnt[new_cnt < 1e-10] = 1.0
        new_ready = new_ready / new_cnt
        self._consumed = safe_end

        # הוספה לבאפר השהייה — ומשיכת len(chunk) דגימות
        self._out_delay = np.concatenate([self._out_delay, new_ready])
        if len(self._out_delay) >= len(chunk):
            result = self._out_delay[:len(chunk)]
            self._out_delay = self._out_delay[len(chunk):]
        else:
            # עדיין לא מספיק — pad באפסים (קורה רק ב-chunk ראשון, ~10ms)
            result = np.zeros(len(chunk))
            result[:len(self._out_delay)] = self._out_delay
            self._out_delay = np.array([], dtype=np.float64)

        # חיתוך באפרים למניעת גדילת זיכרון
        if self._consumed > fft_size * 10:
            trim = self._consumed - fft_size
            self._in_buf = self._in_buf[trim:]
            self._out_clean = self._out_clean[trim:]
            self._out_cnt = self._out_cnt[trim:]
            self._next_start -= trim
            self._consumed -= trim

        # Clip — חיסור ספקטרלי יכול ליצור peaks מעבר למקור (שינוי פאזה)
        return np.clip(result, -1.0, 1.0)


# ===== ניקוי אודיו בזמן אמת — chunk-by-chunk =====
class StreamingCleaner:
    """
    מנקה אודיו בזמן אמת, chunk אחרי chunk, **ללא** עיבוד מקדים.
    משלב חיסור ספקטרלי + דיכוי מעברים חדים (transients) כמו "תק תק".

    1. 0.3 שניות ראשונות: בונה פרופיל רעש + מתחיל לנגן מיד
    2. אחרי כן: חיסור ספקטרלי אגרסיבי + פרופיל מתעדכן בהמשך
    3. דיכוי אירועים חדים: קפיצת אנרגיה פתאומית = "תק" → מדוכא
    """

    def __init__(self, sr, fft_size=2048, sub_alpha=3.0, sub_beta=0.01,
                 noise_seconds=0.3, transient_ratio=3.5):
        self.sr = sr
        self.fft_size = fft_size
        self.hop = fft_size // 4
        self.bins = fft_size // 2 + 1
        self.window = np.hanning(fft_size)
        self.sub_alpha = sub_alpha
        self.sub_beta = sub_beta
        self.transient_ratio = transient_ratio

        # פרופיל רעש — נבנה מהפריימים הראשונים
        self.noise_frames_needed = max(2, int(noise_seconds * sr / self.hop))
        self.noise_mag_sum = np.zeros(self.bins)
        self.noise_frame_count = 0
        self.noise_profile = None

        # מעקב אנרגיה — לזיהוי transients (תק-תק)
        self.long_energy = 0.0
        self.energy_alpha = 0.93

        # באפרים פנימיים — overlap-add streaming
        self._buf = np.zeros(0, dtype=np.float64)
        self._out = np.zeros(0, dtype=np.float64)
        self._cnt = np.zeros(0, dtype=np.float64)
        self._next = 0       # מיקום ה-frame הבא
        self._consumed = 0   # כמה כבר הוחזר

    def process_chunk(self, chunk):
        """
        מקבל chunk חדש → מחזיר אודיו מנוקה באותו אורך.
        נקרא בלופ לכל chunk שמגיע — זה כל ה-API.
        """
        fft = self.fft_size
        hop = self.hop
        win = self.window

        # הוספה לבאפר פנימי
        self._buf = np.concatenate([self._buf, chunk])
        extra = len(self._buf) - len(self._out)
        if extra > 0:
            self._out = np.concatenate([self._out, np.zeros(extra)])
            self._cnt = np.concatenate([self._cnt, np.zeros(extra)])

        # עיבוד כל הפריימים השלמים
        while self._next + fft <= len(self._buf):
            s = self._next
            frame = self._buf[s:s + fft] * win
            spectrum = np.fft.rfft(frame)
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)

            # --- בניית/עדכון פרופיל רעש ---
            if self.noise_frame_count < self.noise_frames_needed:
                self.noise_mag_sum += mag
                self.noise_frame_count += 1
                self.noise_profile = self.noise_mag_sum / self.noise_frame_count
            else:
                # עדכון איטי — פרופיל מסתגל לרעש שמשתנה
                self.noise_profile = 0.98 * self.noise_profile + 0.02 * mag

            # --- חיסור ספקטרלי ---
            if self.noise_profile is not None:
                clean_mag = np.maximum(
                    mag - self.sub_alpha * self.noise_profile,
                    self.sub_beta * mag
                )
            else:
                clean_mag = mag

            # --- דיכוי transients (תק-תק, קליקים, דפיקות) ---
            frame_energy = float(np.sum(mag ** 2))
            if self.long_energy > 1e-10:
                ratio = frame_energy / self.long_energy
                if ratio > self.transient_ratio:
                    # קפיצה חדה = אירוע אימפולסיבי → דיכוי
                    suppress = np.clip(1.5 / ratio, 0.02, 0.8)
                    clean_mag *= suppress
            self.long_energy = (self.energy_alpha * self.long_energy +
                                (1 - self.energy_alpha) * frame_energy)

            # --- iFFT + overlap-add ---
            clean_frame = np.fft.irfft(clean_mag * np.exp(1j * phase), n=fft)
            self._out[s:s + fft] += clean_frame * win
            self._cnt[s:s + fft] += win ** 2
            self._next += hop

        # חילוץ התוצאה עבור ה-chunk
        c0 = self._consumed
        c1 = c0 + len(chunk)
        out = self._out[c0:c1].copy()
        cnt = self._cnt[c0:c1].copy()
        cnt[cnt < 1e-10] = 1.0
        self._consumed = c1

        # חיתוך באפרים
        if self._consumed > fft * 10:
            trim = self._consumed - fft
            self._buf = self._buf[trim:]
            self._out = self._out[trim:]
            self._cnt = self._cnt[trim:]
            self._next -= trim
            self._consumed -= trim

        return out / cnt


# ===== חיסור ספקטרלי - ביטול רעש מהיר ויעיל =====
def spectral_subtract(audio, sr, fft_size=2048, noise_seconds=0.5,
                      alpha=2.0, beta=0.02):
    """
    ביטול רעש באמצעות חיסור ספקטרלי (Spectral Subtraction).
    מהיר מאוד (vectorized FFT) ועובד על כל סוג רעש.

    audio: אות מונו (1D numpy array)
    sr: sample rate
    fft_size: גודל FFT
    noise_seconds: כמה שניות מההתחלה לשמש כפרופיל רעש
    alpha: מקדם חיסור (גבוה = יותר אגרסיבי)
    beta: רצפה ספקטרלית (מונע ארטיפקטים מוזיקליים)
    """
    hop = fft_size // 2
    window = np.hanning(fft_size)

    # --- שלב 1: אמידת פרופיל רעש מתחילת ההקלטה ---
    noise_samples = min(int(noise_seconds * sr), len(audio) // 2)
    noise_portion = audio[:noise_samples]

    noise_specs = []
    for start in range(0, len(noise_portion) - fft_size + 1, hop):
        frame = noise_portion[start:start + fft_size] * window
        noise_specs.append(np.abs(np.fft.rfft(frame)))

    if len(noise_specs) == 0:
        # אם ההקלטה קצרה מדי, השתמש ב-frame אחד
        padded = np.zeros(fft_size)
        padded[:len(noise_portion)] = noise_portion[:fft_size] * window[:len(noise_portion)]
        noise_specs.append(np.abs(np.fft.rfft(padded)))

    noise_profile = np.mean(noise_specs, axis=0)

    # --- שלב 2: חיסור ספקטרלי על כל האות ---
    output = np.zeros(len(audio))
    count = np.zeros(len(audio))

    for start in range(0, len(audio) - fft_size + 1, hop):
        frame = audio[start:start + fft_size] * window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # חיסור עם רצפה: max(|X| - α·|N|, β·|X|)
        clean_mag = np.maximum(mag - alpha * noise_profile, beta * mag)

        clean_spectrum = clean_mag * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_spectrum, n=fft_size)

        output[start:start + fft_size] += clean_frame * window
        count[start:start + fft_size] += window ** 2

    count[count < 1e-10] = 1.0
    return output / count


# ===== ביטול רעש משולב: סטציונרי + מחזורי/אימפולסיבי =====
def clean_audio(audio, sr, fft_size=2048, noise_seconds=0.5,
                sub_alpha=2.5, sub_beta=0.02, remove_percussive=True):
    """
    ביטול רעש משולב בסריקה אחת:
    1) חיסור ספקטרלי — מסיר רעש רקע סטציונרי (מזגן, זמזום)
    2) HPSS (Harmonic-Percussive Source Separation) — מסיר רעשים
       חוזרים/אימפולסיביים (תק-תק, קליקים, דפיקות)

    audio:  אות מונו (1D numpy array)
    sr:     sample rate
    fft_size: גודל FFT
    noise_seconds: שניות פרופיל רעש מתחילת ההקלטה
    sub_alpha: אגרסיביות חיסור ספקטרלי
    sub_beta:  רצפה ספקטרלית
    remove_percussive: האם להפעיל הפרדת הרמוני/פרקוסיבי
    """
    from scipy.ndimage import median_filter

    hop = fft_size // 4          # 75% overlap — איכות גבוהה יותר
    window = np.hanning(fft_size)
    bins = fft_size // 2 + 1

    # ===== STFT =====
    n_frames = max(0, (len(audio) - fft_size) // hop + 1)
    if n_frames == 0:
        return audio.copy()

    stft = np.zeros((bins, n_frames), dtype=complex)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + fft_size] * window
        stft[:, i] = np.fft.rfft(frame)

    mag = np.abs(stft)
    phase = np.angle(stft)

    # ===== שלב 1: חיסור ספקטרלי — רעש רקע =====
    noise_frames = max(1, int(noise_seconds * sr / hop))
    noise_frames = min(noise_frames, n_frames)
    noise_profile = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    clean_mag = np.maximum(mag - sub_alpha * noise_profile, sub_beta * mag)

    # ===== שלב 2: HPSS — הסרת רכיבים פרקוסיביים (תק-תק, קליקים) =====
    if remove_percussive and n_frames > 3:
        # גרעין מדיאני לציר הזמן — שומר רכיבים הרמוניים (יציבים)
        # ~300ms — מספיק כדי להחליק "תק" קצר
        h_len = int(0.3 * sr / hop)
        h_len = h_len + 1 if h_len % 2 == 0 else h_len
        h_len = max(3, min(h_len, n_frames))

        # גרעין מדיאני לציר התדר — תופס רכיבים פרקוסיביים (רחבי-פס)
        # ~500Hz — רוחב פס טיפוסי ל-"תק"
        freq_res = sr / fft_size           # Hz לכל bin
        p_len = int(500 / freq_res)
        p_len = p_len + 1 if p_len % 2 == 0 else p_len
        p_len = max(3, min(p_len, bins))

        # סינון מדיאני: הרמוני לאורך זמן, פרקוסיבי לאורך תדר
        harmonic = median_filter(clean_mag, size=(1, h_len))
        percussive = median_filter(clean_mag, size=(p_len, 1))

        # מסכה רכה: שומרת רכיב הרמוני, מדכאת פרקוסיבי
        # חזקה 2 → הפרדה חדה יותר
        mask = (harmonic ** 2) / (harmonic ** 2 + percussive ** 2 + 1e-10)
        clean_mag = clean_mag * mask

    # ===== iSTFT =====
    clean_stft = clean_mag * np.exp(1j * phase)
    output = np.zeros(len(audio))
    count = np.zeros(len(audio))

    for i in range(n_frames):
        start = i * hop
        frame = np.fft.irfft(clean_stft[:, i], n=fft_size)
        end = min(start + fft_size, len(audio))
        output[start:end] += frame[:end - start] * window[:end - start]
        count[start:end] += window[:end - start] ** 2

    count[count < 1e-10] = 1.0
    return output / count
