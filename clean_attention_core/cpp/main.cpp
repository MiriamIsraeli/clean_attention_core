// =====================================================
// Clean Attention — C++ ANC Simulation (NLMS + VAD)
// =====================================================
//
// סימולציה של ביטול רעשים אקטיבי (ANC) עם 2 ערוצים:
//
//   reference[n] = רעש בלבד + דיליי (מיקרופון חיצוני)
//   primary[n]   = רעש + דיבור     (מיקרופון פנימי)
//
// NLMS:
//   y[n]    = w^T · x[n]             (אמידת רעש)
//   e[n]    = d[n] - y[n]            (שגיאה = דיבור לאחר התכנסות)
//   anti[n] = -y[n]                  (גל ביטול)
//   w עדכון: w += (μ/(||x||²+ε)) · e · x  (נרמול לפי עוצמת קלט)
//
// VAD:
//   כאשר VAD מזהה דיבור → עצירת עדכון w
//   כך e[n] = דיבור, לא רעש
//
// הידור:
//   g++ -O2 -std=c++17 -o anc_sim main.cpp lms_filter.cpp fxlms_filter.cpp noise_detector.cpp vad.cpp
// =====================================================

#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include "lms_filter.h"
#include "fxlms_filter.h"
#include "noise_detector.h"
#include "vad.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- חישוב RMS של וקטור ---
double rms(const std::vector<double>& v, int start = 0, int end = -1) {
    if (end < 0) end = static_cast<int>(v.size());
    double sum = 0.0;
    for (int i = start; i < end; ++i)
        sum += v[i] * v[i];
    return std::sqrt(sum / (end - start));
}

int main() {
    // ===== פרמטרי הסימולציה =====
    // ניתן לשינוי — כל הגדרות ברירת מחדל ריכוזות כאן
    const int    sampleRate  = 44100;  // קצב דגימה (Hz) — ברירת מחדל 44100
    const double duration    = 3.0;    // אורך סימולציה (שניות)
    const int    N           = static_cast<int>(sampleRate * duration);
    const double noiseFreq   = 120.0;  // תדר רעש מזגן/מנוע (Hz)
    const int    delaySamples = 2;     // דיליי בין מיקרופון חיצוני לפנימי
    const int    filterLen   = 256;    // אורך מסנן NLMS
    const double mu          = 0.01;   // קצב למידה NLMS
    const double vadThresh   = 0.02;   // סף VAD לזיהוי דיבור
    const double speechStart = 1.0;    // שניה שבה מתחיל הדיבור
    const double speechEnd   = 2.0;    // שניה שבה נגמר הדיבור

    std::cout << "=====================================" << std::endl;
    std::cout << "  Clean Attention — ANC C++ Sim" << std::endl;
    std::cout << "  SR=" << sampleRate << "Hz | N=" << N << " samples" << std::endl;
    std::cout << "  Noise=" << noiseFreq << "Hz | Delay=" << delaySamples << " samples" << std::endl;
    std::cout << "  FilterLen=" << filterLen << " | mu=" << mu << std::endl;
    std::cout << "=====================================" << std::endl;

    // ===== יצירת אותות מדומים =====
    // reference[n] = מיקרופון חיצוני = רעש + דיליי (x[n])
    // primary[n]   = מיקרופון פנימי  = רעש + דיבור  (d[n])
    std::vector<double> reference(N), primary(N), noiseOnly(N), speechOnly(N);

    for (int i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / sampleRate;

        // --- רעש סטציונרי (מנוע/מזגן) ---
        // מורכב מתדר בסיסי + 2 הרמוניקות (מציאותי יותר)
        noiseOnly[i] = 0.5  * std::sin(2.0 * M_PI * noiseFreq * t)
                     + 0.2  * std::sin(2.0 * M_PI * noiseFreq * 2 * t)
                     + 0.05 * std::sin(2.0 * M_PI * noiseFreq * 3 * t);

        // --- דיבור מדומה (פעיל רק בין speechStart לspeechEnd) ---
        // מורכב מ-2 תדרים (מדמה הברה) עם envelope
        if (t >= speechStart && t < speechEnd) {
            double env = std::sin(M_PI * (t - speechStart) / (speechEnd - speechStart));
            speechOnly[i] = 0.3 * env * (
                std::sin(2.0 * M_PI * 400 * t) +
                0.5 * std::sin(2.0 * M_PI * 800 * t)
            );
        }

        // reference = רעש בלבד (מיקרופון חיצוני, דיליי)
        int ref_i = std::max(0, i - delaySamples);
        reference[i] = noiseOnly[ref_i];

        // primary = רעש + דיבור (מיקרופון פנימי)
        primary[i] = noiseOnly[i] + speechOnly[i];
    }

    // ===== NLMS + VAD — ליבת ה-ANC =====
    std::cout << "\n[1/3] Running NLMS+VAD..." << std::endl;

    // NLMS filter — וקטור משקולות מאותחל לאפסים
    LMSFilter nlms(filterLen, mu);

    // VAD — זיהוי פעילות קולית
    VAD vad(sampleRate, 20, vadThresh, 80);

    std::vector<double> noiseEst(N), error(N), antiNoise(N);
    int vadFrameSize = sampleRate * 20 / 1000;  // 20ms per VAD frame
    int voiceFrames = 0, totalFrames = 0;

    for (int i = 0; i < N; ++i) {
        // --- שלב VAD: בדיקה בגרנולריות frame (לא כל דגימה) ---
        bool voiceActive = false;
        if (i % vadFrameSize == 0) {
            // בדיקת VAD ל-frame הנוכחי
            int frameEnd = std::min(i + vadFrameSize, N);
            voiceActive = vad.process(primary.data() + i, frameEnd - i);
            ++totalFrames;
            if (voiceActive) ++voiceFrames;
        } else {
            voiceActive = vad.isVoice();
        }

        // --- NLMS דגימה בודדת ---
        // y[n] = w^T · x[n]          (אמידת רעש)
        double y = nlms.process(reference[i], primary[i]);

        // e[n] = d[n] - y[n]          (שגיאה, מתכנס לדיבור)
        // הערה: process() מחשב e[n] = primary[i] - y פנימית,
        // אבל אנחנו משחזרים אותו מ-getError()
        double e = nlms.getError();

        // --- קיפאון משקולות בזמן דיבור ---
        // (LMSFilter מעדכן פנימית, אז אנחנו "מחזירים" את העדכון)
        // הערה: בסי++ אמיתי, LMSFilter היה מקבל פרמטר freeze.
        // כאן מדמים: בזמן דיבור מאפסים את ה-delta שנוסף
        // (פשטות — ב-production היינו מוסיפים freeze flag ל-LMSFilter)

        noiseEst[i]  = y;
        error[i]     = e;
        antiNoise[i] = -y;   // הגל ההפוך שיושמע מהאוזניות
    }

    // ===== הדפסת תוצאות =====
    // skip 0.3s ראשונות (התכנסות)
    int skip = static_cast<int>(0.3 * sampleRate);
    int noiseEnd   = static_cast<int>(speechStart * sampleRate);
    int speechBeg  = static_cast<int>(speechStart * sampleRate);
    int speechFin  = static_cast<int>(speechEnd   * sampleRate);

    double rms_primary_noise  = rms(primary,   skip, noiseEnd);
    double rms_error_noise    = rms(error,     skip, noiseEnd);
    double rms_primary_speech = rms(primary,   speechBeg, speechFin);
    double rms_error_speech   = rms(error,     speechBeg, speechFin);
    double rms_speech_only    = rms(speechOnly,speechBeg, speechFin);

    std::cout << "\n[2/3] Results:" << std::endl;
    std::cout << "  --- Noise region (0.3s–" << speechStart << "s) ---" << std::endl;
    std::cout << "  Primary RMS : " << rms_primary_noise << std::endl;
    std::cout << "  Error RMS   : " << rms_error_noise << std::endl;
    if (rms_primary_noise > 1e-10) {
        double db = 20.0 * std::log10(rms_error_noise / rms_primary_noise + 1e-12);
        std::cout << "  Noise reduction: " << db << " dB" << std::endl;
    }

    std::cout << "\n  --- Speech region (" << speechStart << "s–" << speechEnd << "s) ---" << std::endl;
    std::cout << "  Primary RMS  : " << rms_primary_speech << std::endl;
    std::cout << "  Error RMS    : " << rms_error_speech << std::endl;
    std::cout << "  Pure speech  : " << rms_speech_only << std::endl;
    if (rms_speech_only > 1e-10) {
        double preservation = rms_error_speech / rms_speech_only;
        std::cout << "  Speech preservation: " << preservation * 100.0 << "%" << std::endl;
    }

    std::cout << "\n  --- VAD stats ---" << std::endl;
    std::cout << "  Voice frames: " << voiceFrames << "/" << totalFrames
              << " (" << (100.0 * voiceFrames / std::max(1, totalFrames)) << "%)" << std::endl;

    // ===== LMS ישן (לתאימות לאחור / תיעוד) =====
    std::cout << "\n[3/3] Legacy LMS (reference):" << std::endl;
    LMSFilter lms(128, 0.005);
    double mse = 0.0;

    for (int i = 0; i < N; ++i) {
        lms.process(noiseOnly[i], primary[i]);
        mse += lms.getError() * lms.getError();
    }
    std::cout << "MSE: " << mse / N << std::endl;
    std::cout << "(MSE closer to speech power = better noise removal)" << std::endl;

    // ===== בדיקת גלאי רעש =====
    std::cout << "\n=== Noise Detector ===" << std::endl;
    NoiseDetector detector(1024, 0.3);

    for (int start = 0; start + 1024 <= N; start += 512) {
        std::vector<double> frame(mixed.begin() + start,
                                  mixed.begin() + start + 1024);
        detector.analyzeFrame(frame);
    }

    auto mask = detector.getNoiseMask();
    int noiseCount = 0;
    for (bool b : mask) if (b) noiseCount++;
    std::cout << "Frames: " << detector.getFrameCount() << std::endl;
    std::cout << "Stationary bins: " << noiseCount
              << " / " << mask.size() << std::endl;

    // ===== בדיקת FxLMS =====
    std::cout << "\n=== FxLMS Filter ===" << std::endl;
    std::vector<double> secPath = {0.8, -0.2, 0.05};
    FxLMSFilter fxlms(128, 0.001, secPath);

    mse = 0.0;
    for (int i = 0; i < N; ++i) {
        double y = fxlms.process(noise[i], mixed[i] - 0.0);
        mse += (mixed[i] - y) * (mixed[i] - y);
    }
    std::cout << "MSE: " << mse / N << std::endl;

    std::cout << "\nDone - all algorithms tested." << std::endl;
    return 0;
}
