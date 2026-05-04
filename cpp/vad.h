#pragma once
#include <vector>
#include <cmath>

// ===== VAD — Voice Activity Detection =====
// זיהוי פעילות קולית בזמן אמת
//
// עקרון הפעולה:
//   מחשב אנרגיה (RMS) של כל חלון זמן קצר.
//   אם האנרגיה עולה על הסף (threshold) — דיבור מזוהה.
//
//   EMA אסימטרי:
//     attack  (עלייה מהירה)  — מזהה תחילת דיבור תוך ~2 frames
//     release (ירידה איטית) — לא "מפספס" סוף מילה
//
//   hold_frames: שומר על מצב VOICE כמה frames לאחר שהאנרגיה ירדה
//     מונע חיתוך של עיצורים שקטים כמו p/t/k
//
// שימוש טיפוסי עם NLMSFilter:
//   כאשר is_voice() == true:
//     → NLMSFilter מדלג על עדכון המשקולות
//     → e[n] = primary[n] - y[n]  (שגיאה = דיבור + שאריות)
//   כאשר is_voice() == false:
//     → NLMSFilter מעדכן משקולות
//     → e[n] מתכנס לדיבור בלבד לאחר התכנסות
class VAD {
public:
    // sr           : קצב דגימה (Hz)
    // frameMs      : אורך חלון VAD במילישניות
    // threshold    : סף אנרגיה RMS (0.0–1.0), ברירת מחדל 0.02
    // holdMs       : כמה ms לשמור VOICE לאחר ירידת אנרגיה
    VAD(int sr = 44100,
        int frameMs = 20,
        double threshold = 0.02,
        int holdMs = 80);

    // עדכון VAD עם דגימות חדשות.
    // מחזיר true אם יש דיבור פעיל.
    // chunk: מצביע לדגימות, count: מספר הדגימות
    bool process(const double* chunk, int count);

    // מצב נוכחי
    bool isVoice() const { return isVoice_; }

    // האנרגיה הנוכחית (לדיבוג/ויזואליזציה)
    double energyEma() const { return energyEma_; }

    // איפוס
    void reset();

private:
    double threshold_;     // סף אנרגיה
    int holdFrames_;       // frames להחזקת VOICE
    int holdCounter_;      // מונה frames שנשאר ב-VOICE
    double energyEma_;     // EMA של אנרגיה
    double alphaAttack_;   // alpha לעלייה (מהיר)
    double alphaRelease_;  // alpha לירידה (איטי)
    bool isVoice_;         // מצב נוכחי
};
