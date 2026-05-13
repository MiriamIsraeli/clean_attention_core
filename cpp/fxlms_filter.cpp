-#include "fxlms_filter.h"
#include <algorithm>
#include <numeric>

// --- אתחול FxLMS עם נתיב משני ---
FxLMSFilter::FxLMSFilter(int filterLength, double stepSize,
                          const std=::vector<double>& secondaryPath)
    : N_(filterLength), mu_(stepSize), w_(filterLength, 0.0),
      xBuf_(filterLength, 0.0), xfBuf_(filterLength, 0.0),
      secPath_(secondaryPath) {}

// --- עיבוד דגימה ---
// ההבדל מ-LMS: הקלט מסונן דרך מודל הנתיב המשני לפני עדכון
// זה מבטיח שהגל הנגדי יגיע בפאזה הנכונה לנקודת הביטול
double FxLMSFilter::process(double reference, double error) {
    // הכנסת דגימה לבאפר הקלט
    for (int i = N_ - 1; i > 0; --i)
        xBuf_[i] = xBuf_[i - 1];
    xBuf_[0] = reference;

    // סינון הקלט דרך הנתיב המשני: x'[n] = S_hat * x[n]
    double xf = 0.0;
    int spLen = static_cast<int>(secPath_.size());
    for (int i = 0; i < spLen && i < N_; ++i)
        xf += secPath_[i] * xBuf_[i];

    // הכנסת הקלט המסונן לבאפר
    for (int i = N_ - 1; i > 0; --i)
        xfBuf_[i] = xfBuf_[i - 1];
    xfBuf_[0] = xf;

    // חישוב אות הביטול: y = w^T * x
    double y = std::inner_product(w_.begin(), w_.end(), xBuf_.begin(), 0.0);

    // עדכון משקולות לפי הקלט המסונן: w += μ * e * x'
    for (int i = 0; i < N_; ++i)
        w_[i] += mu_ * error * xfBuf_[i];

    return y;
}

// --- איפוס ---
void FxLMSFilter::reset() {
    std::fill(w_.begin(), w_.end(), 0.0);
    std::fill(xBuf_.begin(), xBuf_.end(), 0.0);
    std::fill(xfBuf_.begin(), xfBuf_.end(), 0.0);
}
