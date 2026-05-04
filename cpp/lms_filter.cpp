#include "lms_filter.h"
#include <algorithm>
#include <numeric>

// --- אתחול המסנן: איפוס משקולות ובאפר ---
LMSFilter::LMSFilter(int filterLength, double stepSize)
    : N_(filterLength), mu_(stepSize), w_(filterLength, 0.0),
      x_(filterLength, 0.0), lastError_(0.0) {}

// --- עיבוד דגימה בודדת ---
// 1. הכנסת הדגימה החדשה לבאפר (הזזה)
// 2. חישוב פלט: מכפלה פנימית של משקולות וקלט
// 3. חישוב שגיאה: ההפרש בין האות הרצוי לפלט
// 4. עדכון משקולות לפי כלל LMS: w += μ * e * x
double LMSFilter::process(double reference, double desired) {
    // הזזת באפר הקלט והכנסת דגימה חדשה
    for (int i = N_ - 1; i > 0; --i)
        x_[i] = x_[i - 1];
    x_[0] = reference;

    // מכפלה פנימית: y = Σ(w[i] * x[i])
    double y = std::inner_product(w_.begin(), w_.end(), x_.begin(), 0.0);

    // שגיאה = רצוי - פלט
    lastError_ = desired - y;

    // עדכון משקולות (כלל הלמידה של LMS)
    for (int i = 0; i < N_; ++i)
        w_[i] += mu_ * lastError_ * x_[i];

    return y;  // אמידת הרעש
}

// --- איפוס המסנן למצב התחלתי ---
void LMSFilter::reset() {
    std::fill(w_.begin(), w_.end(), 0.0);
    std::fill(x_.begin(), x_.end(), 0.0);
    lastError_ = 0.0;
}
