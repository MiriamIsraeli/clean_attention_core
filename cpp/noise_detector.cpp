#include "noise_detector.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- אתחול גלאי הרעש ---
NoiseDetector::NoiseDetector(int fftSize, double stabilityThreshold)
    : fftSize_(fftSize), threshold_(stabilityThreshold), frameCount_(0),
      sumMag_(fftSize / 2 + 1, 0.0), sumMagSq_(fftSize / 2 + 1, 0.0) {}

// --- ניתוח מסגרת שמע ---
// 1. הפעלת חלון Hanning למניעת דליפה ספקטרלית
// 2. ביצוע FFT לקבלת ספקטרום התדרים
// 3. עדכון ממוצע ושונות העוצמה לכל תדר
void NoiseDetector::analyzeFrame(const std::vector<double>& frame) {
    std::vector<std::complex<double>> data(fftSize_);
    for (int i = 0; i < fftSize_; ++i) {
        double window = 0.5 * (1.0 - cos(2.0 * M_PI * i / (fftSize_ - 1)));
        double val = (i < (int)frame.size()) ? frame[i] : 0.0;
        data[i] = std::complex<double>(val * window, 0.0);
    }

    fft(data);

    int halfN = fftSize_ / 2 + 1;
    for (int i = 0; i < halfN; ++i) {
        double mag = std::abs(data[i]);
        sumMag_[i] += mag;
        sumMagSq_[i] += mag * mag;
    }
    frameCount_++;
}

// --- מסכת רעש סטציונרי ---
// מקדם שונות (CV) נמוך = תדר יציב = רעש מחזורי
// תדרים עם CV מתחת לסף מסומנים כרעש
std::vector<bool> NoiseDetector::getNoiseMask() const {
    int halfN = fftSize_ / 2 + 1;
    std::vector<bool> mask(halfN, false);
    if (frameCount_ < 2) return mask;

    for (int i = 0; i < halfN; ++i) {
        double mean = sumMag_[i] / frameCount_;
        double variance = sumMagSq_[i] / frameCount_ - mean * mean;
        double stddev = (variance > 0) ? sqrt(variance) : 0.0;
        double cv = (mean > 1e-10) ? (stddev / mean) : 1e10;
        mask[i] = (cv < threshold_) && (mean > 1e-8);
    }
    return mask;
}

// --- ספקטרום רעש מוערך ---
std::vector<double> NoiseDetector::getNoiseSpectrum() const {
    int halfN = fftSize_ / 2 + 1;
    std::vector<double> spectrum(halfN, 0.0);
    if (frameCount_ == 0) return spectrum;

    auto mask = getNoiseMask();
    for (int i = 0; i < halfN; ++i) {
        if (mask[i])
            spectrum[i] = sumMag_[i] / frameCount_;
    }
    return spectrum;
}

// ===== FFT - Cooley-Tukey Radix-2 =====
// התמרת פורייה מהירה לניתוח תדרים
// דרישה: אורך הקלט חייב להיות חזקה של 2
void NoiseDetector::fft(std::vector<std::complex<double>>& data) {
    int n = static_cast<int>(data.size());
    if (n <= 1) return;

    // שלב 1: סידור ביטים הפוך (bit-reversal permutation)
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
""
    // שלב 2: חישוב FFT איטרטיבי (butterfly)
    for (int len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / len;
        std::complex<double> wn(cos(angle), sin(angle));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                auto u = data[i + j];
                auto v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wn;
            }
        }
    }
}
