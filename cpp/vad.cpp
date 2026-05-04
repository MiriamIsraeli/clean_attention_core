#include "vad.h"
#include <cmath>
#include <algorithm>

// --- אתחול VAD ---
// מחשב גדלים מ-sr ו-ms פרמטרים
VAD::VAD(int sr, int frameMs, double threshold, int holdMs)
    : threshold_(threshold),
      holdCounter_(0),
      energyEma_(0.0),
      isVoice_(false)
{
    // מספר frames להחזקת VOICE: holdMs / frameMs
    // לפחות frame אחד
    holdFrames_ = std::max(1, holdMs / std::max(1, frameMs));

    // alpha לעלייה: תגובה מהירה ~2 frames
    // ערך גבוה = תגובה מהירה יותר
    alphaAttack_  = 0.6;

    // alpha לירידה: שחרור איטי ~20 frames
    // ערך נמוך = שחרור איטי יותר
    alphaRelease_ = 0.05;
}

// --- עיבוד chunk של דגימות ---
// האלגוריתם:
//   1. חישוב RMS של ה-chunk: sqrt(mean(x²))
//   2. EMA אסימטרי — עולה מהר כשיש אנרגיה, יורד לאט כשאין
//   3. אם EMA > threshold → VOICE + איפוס hold
//   4. אחרת → hold countdown
bool VAD::process(const double* chunk, int count) {
    if (count <= 0) return isVoice_;

    // --- שלב 1: חישוב RMS ---
    // RMS = sqrt(1/N * Σx²) — מדד אנרגיה שלא מושפע מסימן
    double sumSq = 0.0;
    for (int i = 0; i < count; ++i)
        sumSq += chunk[i] * chunk[i];
    double rms = std::sqrt(sumSq / count);

    // --- שלב 2: EMA אסימטרי ---
    // עלייה: alpha גבוה → מגיב מהר לדיבור חדש
    // ירידה: alpha נמוך → לא נחתך אמצע מילה
    if (rms > energyEma_) {
        // אנרגיה עולה — attack מהיר
        energyEma_ = alphaAttack_ * rms + (1.0 - alphaAttack_) * energyEma_;
    } else {
        // אנרגיה יורדת — release איטי
        energyEma_ = alphaRelease_ * rms + (1.0 - alphaRelease_) * energyEma_;
    }

    // --- שלב 3: קבלת החלטה ---
    if (energyEma_ >= threshold_) {
        // מעל הסף → דיבור ברור
        // איפוס מונה ה-hold (נספור מחדש מהנקודה הזו)
        holdCounter_ = holdFrames_;
        isVoice_ = true;
    } else if (holdCounter_ > 0) {
        // מתחת לסף אך עדיין ב"זנב" של דיבור
        --holdCounter_;
        isVoice_ = true;
    } else {
        // שתיקה אמיתית
        isVoice_ = false;
    }

    return isVoice_;
}

// --- איפוס למצב התחלתי ---
void VAD::reset() {
    energyEma_   = 0.0;
    holdCounter_ = 0;
    isVoice_     = false;
}
