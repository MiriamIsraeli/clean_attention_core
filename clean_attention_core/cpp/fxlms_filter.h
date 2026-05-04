#pragma once
#include <vector>

// ===== מסנן FxLMS (Filtered-x LMS) =====
// הרחבה של LMS עבור ביטול רעשים אקטיבי (ANC)
// משקלל את הנתיב המשני (מהרמקול לאוזן) לדיוק מרבי
class FxLMSFilter {
public:
    // filterLength: אורך המסנן
    // stepSize: קצב למידה
    // secondaryPath: מודל הנתיב המשני (תגובת החלל האקוסטי)
    FxLMSFilter(int filterLength, double stepSize,
                const std::vector<double>& secondaryPath);

    // עיבוד דגימה: מקבל אות ייחוס ושגיאה, מחזיר אות ביטול
    double process(double reference, double error);

    const std::vector<double>& getWeights() const { return w_; }
    void reset();

private:
    int N_;
    double mu_;
    std::vector<double> w_;       // משקולות המסנן
    std::vector<double> xBuf_;    // באפר קלט
    std::vector<double> xfBuf_;   // באפר קלט מסונן
    std::vector<double> secPath_; // מודל נתיב משני
};
