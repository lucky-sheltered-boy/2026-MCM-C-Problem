# Data Dictionary & Feature Engineering Report (数据字典与特征工程报告)

This document details the advanced statistical features constructed during the Data Preprocessing Phase. These features serve to transform raw "judge scores" into quantifiable "competitiveness metrics" and "fan influence proxy metrics."

---

## 1. Weekly Performance Metrics (实时竞技指标)
These metrics capture specific performance in each week (Week $w$), normalized to eliminate noise caused by varying seasons (e.g., changes in the number of judges).

### 1.1 `week{w}_total_score` (Weekly Total Score)
*   **Definition**: The sum of scores given by all present judges in Week $w$.
*   **Calculation Logic**:
    $$ S_{i,w} = \sum_{j=1}^{K_w} \text{Score}_{i,w,j} $$
    Where $K_w$ is the number of judges for that week (usually 3 or 4). If a judge is absent (N/A), they are ignored. If a contestant is eliminated, the score is 0.

### 1.2 `week{w}_percent_score` (Weekly Score Percentage)
*   **Definition**: The proportion of points received relative to the theoretical maximum. This is the core standardized metric for cross-season comparison.
*   **Calculation Logic**:
    $$ P_{i,w} = \frac{S_{i,w}}{K_w \times 10} $$
    *   **Range**: $[0, 1]$. For example, if 3 judges give a total of 27 points, the percentage is $27/30 = 0.9$.

### 1.3 `week{w}_judge_rank` (Weekly Judge Rank)
*   **Definition**: The ranking of the contestant based on judge scores among all active contestants in that week of the season.
*   **Calculation Logic**: Uses "Standard Competition Ranking" ("1224" method).
    *   If two contestants tie for 1st, the next is ranked 3rd.
    *   **Significance**: Directly determines pressure in the "Rank Method." A **lower** rank value indicates a **higher** judge evaluation.

---

## 2. Season-Aggregated Metrics (赛季综合实力)
These metrics build a comprehensive profile of a contestant's strength.

### 2.1 `avg_weekly_score` (Average Output Intensity)
*   **Definition**: The average weekly total score during the contestant's active period.
*   **Calculation Logic**:
    $$ \bar{S}_i = \frac{1}{|W_{active}|} \sum_{w \in W_{active}} S_{i,w} $$
    Only counts active weeks where score > 0 to avoid "0 scores" from elimination weeks dragging down the average.

### 2.2 `score_std_dev` (Performance Stability)
*   **Definition**: The standard deviation of the contestant's weekly scores.
*   **Significance**:
    *   **Low Value**: Consistent performance (e.g., professional athletes).
    *   **High Value**: Volatile performance ("wildcard" contestants), often leading to controversy.

### 2.3 `avg_judge_rank` (Average Dominance)
*   **Definition**: The average ranking of the contestant throughout the season.
*   **Significance**: Better reflects relative standing in the contestant pool than raw scores.

---

## 3. Contextual & External Effects (外部效应指标)
Quantifies influence from non-contestant factors.

### 3.1 `partner_avg_placement` (Partner Dividend Coefficient)
*   **Definition**: The average final placement of the professional partner across all their historical seasons.
*   **Calculation Logic**:
    $$ \text{PartnerSkill}_p = \text{Mean}(\text{FinalPlacement}_{\text{all terms for } p}) $$
*   **Interpretation**:
    *   **Lower Value** (e.g., ~2.0 for Derek Hough): Indicates a top-tier partner; the contestant likely benefited from a "great partner" dividend.
    *   **Higher Value**: Indicates a partner with an average track record.

### 3.2 `industry_prevalence` (Industry Heat)
*   **Definition**: The frequency of the contestant's industry (e.g., Actor, Athlete) appearing in historical data. Used to measure if certain industries are "casting favorites."

---

## 4. Fan Base Proxy Variables (观众影响力反代理)
**(Core Innovation of This Model)**
Since actual fan votes are unknown, we reverse-engineer the fan base size using "survival under adversity."

### 4.1 `total_fan_saves_bottom1` (Escapes from Death)
*   **Definition**: The number of times a contestant survived elimination despite being ranked **last (Bottom 1)** by judges.
*   **Significance**: A signal of extreme fan effect. Only massive fan voting can offset the disadvantage of being last in judge scores.

### 4.2 `total_fan_saves_bottom2` (Survival in Extreme Danger)
*   **Definition**: The number of times a contestant survived despite being in the **bottom two (Bottom 2)** of judge rankings.
*   **Significance**: A signal of strong fan effect.

### 4.3 `total_fan_saves_bottom3` (Survival in Danger Zone)
*   **Definition**: The number of times a contestant survived despite being in the **bottom three (Bottom 3)** of judge rankings.
*   **Significance**: Indicates moderate fan effect or luck (since bottom three is often relatively safe).

---

## Summary: Why Are These Columns Crucial?

In the upcoming modeling (Problem C):
1.  **Bayesian Vote Reconstruction**: `week{w}_judge_rank` will serve as our **Observed Variable** for constructing the likelihood function.
2.  **Controversy Analysis**: We will filter contestants with `total_fan_saves_bottom1` > 0 (e.g., Bobby Bones) as key cases for "High Discrepancy" analysis.
3.  **Model Prediction**: `partner_avg_placement` and `avg_judge_rank` will be input features for regression models to quantify the contribution of "skill" vs. "background" to the final ranking.
