#pragma once
#include <vector>
#include <complex>

// ===== גלאי רעש סטציונרי =====
// מזהה תדרים יציבים (רעש מחזורי קבוע) באמצעות ניתוח FFT
// תדרים עם שונות נמוכה לאורך זמן = רעש סטציונרי
class NoiseDetector {
public:
    // fftSize: גודל חלון הניתוח (חזקת 2)
    // stabilityThreshold: סף יציבות - ערך נמוך = רגישות גבוהה
    NoiseDetector(int fftSize = 1024, double stabilityThreshold = 0.3);

    // ניתוח מסגרת שמע ועדכון סטטיסטיקות
    void analyzeFrame(const std::vector<double>& frame);

    // מסכת רעש: true = תדר סטציונרי (רעש שיש לבטל)
    std::vector<bool> getNoiseMask() const;

    // ספקטרום הרעש המוערך
    std::vector<double> getNoiseSpectrum() const;

    int getFrameCount() const { return frameCount_; }

private:
    int fftSize_;
    double threshold_;
    int frameCount_;
    std::vector<double> sumMag_;
    std::vector<double> sumMagSq_;

    // FFT - Cooley-Tukey Radix-2
    static void fft(std::vector<std::complex<double>>& data);
};
