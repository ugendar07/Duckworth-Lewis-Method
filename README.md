# DLS Method for Run Production Functions

## Overview
This repository contains an implementation of the DLS (Duckworth-Lewis Method) method for finding the best-fit run production functions using the first innings data alone. The objective is to model the relationship between run production, wickets-in-hand (W), and overs-to-go (u) in a cricket match.

## Methodology
We utilize the following model for run production:
\[ Z_{(u,w)} = Z0_{(w)}[1 - \exp\left(-\frac{Lu}{Z0_{(w)}}\right)] \]

Where:
- \( Z_{(u,w)} \) represents the run production at a given point in time.
- \( Z0_{(w)} \) represents the initial run production with respect to wickets-in-hand.
- \( L \) is the rate parameter.
- \( u \) represents the overs-to-go.
- \( w \) represents the wickets-in-hand.

We employ the sum of squared errors loss function, summed across overs, wickets, and data points for those overs and wickets.

## Notes
- This implementation focuses solely on the first innings data.
- Adjustments may be needed for different datasets or specific requirements.

## Contact
For any inquiries or suggestions, please contact ugendar07@gmail.com.
