#pragma once
#include <vector>
#include <cmath>
#include <numeric>

// ===== מסנן LMS (Least Mean Squares) =====
// אלגוריתם סינון אדפטיבי — ליבת ביטול הרעשים
// לומד את דפוס הרעש ומייצר אות ביטול (גל נגדי)
//
// כלל העדכון (LMS):  w[n+1] = w[n] + μ · e[n] · x[n]
// כלל העדכון (NLMS): w[n+1] = w[n] + μ · e[n] · x[n] / (||x||² + ε)
//
// LMS: פשוט, מהיר, אך רגיש לגודל הקלט (μ תלוי בעוצמה)
// NLMS: יציב יותר — מנרמל לפי עוצמת x[n]
class LMSFilter {
public:
    // filterLength: מספר המשקולות (אורך המסנן)
    // stepSize: קצב הלמידה (μ) - קובע מהירות ההתכנסות
    LMSFilter(int filterLength, double stepSize);

    // עיבוד דגימה בודדת בזמן אמת
    // reference: אות הרעש מהמיקרופון המייחס
    // desired: האות המעורב (רעש + דיבור)
    // מחזיר: אמידת הרעש (y) - הגל הנגדי הוא מינוס y
    double process(double reference, double desired);

    const std::vector<double>& getWeights() const { return w_; }
    double getError() const { return lastError_; }
    void reset();

private:
    int N_;
    double mu_;
    std::vector<double> w_;  // וקטור משקולות
    std::vector<double> x_;  // באפר דגימות
    double lastError_;
};
