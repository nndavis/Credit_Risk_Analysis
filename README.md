# Credit_Risk_Analysis
 
## Overview of the Analysis

This analysis uses machine learning to compare credit risk data: good loans against risky loans. Good loans heavily outweigh the number of risky loans, so to start I used oversampling and undersampling machine learning from the scikit-learn library. After that, I used a SMOTEENN algorithm that combines both oversampling and undersampling. The plus side to this is that it creates synthetic data points rather than create duplicate data points. This analysis also uses BalancedRandomForestClassifier to reduce bias, and EasyEnsembleClassifier to predict credit risk. These are machine learning models that come from imblearn.ensemble a package related to scikit-learn. After the models and reports have been generated it is time to review the results.

## Results
- Naive oversampling
  -
  - The accuracy score for naive oversampling is 0.646.
  - The high risk to low risk precision is 0.01 to 1.00.
  - The high risk and low risk recall scores are 0.71 and 0.58.
  - The accuracy score is too low to be viable for testing. The high risk precicision is bad, but for this analysis it will be best to look at recall to correctly evaluate if a customer is high risk or low risk. The recall score is 0.71 for high risk which is not the greatest, as 29% of high risk customers will not be predicted correctly.
  
<img width="337" alt="image" src="https://user-images.githubusercontent.com/104074135/199358854-07e00a75-8ab6-46fe-b52f-e1430397c8d2.png">

<img width="493" alt="image" src="https://user-images.githubusercontent.com/104074135/199358914-a3b7e39f-b387-4261-9fd3-f9d0d076f4e9.png">

- SMOTE oversampling
  -
  - The accuracy score for SMOTE oversampling is 0.659.
  - The high risk to low risk precision is 0.01 to 1.00.
  - The high risk and low risk recall scores are 0.63 and 0.68.
  - The accuracy score is slightly better with SMOTE than Naive oversampling, but it is still not viable. The precision stays the same, but it is also not the most important statistic here. The recall lowered for high risk but increased for the low. They are also closer in value. I would say SMOTE is slightly better than naive oversampling, but still not viable.
  
<img width="362" alt="image" src="https://user-images.githubusercontent.com/104074135/199359036-a9b98912-9801-4146-a28c-885b134d7f3a.png">

<img width="488" alt="image" src="https://user-images.githubusercontent.com/104074135/199359170-71292b88-ca6a-41b1-9005-289b874b14e0.png">

- Undersampling
  -
  - The accuracy score for undersampling is 0.544.
  - The high risk to low risk precision is 0.01 to 1.00.
  - The high risk and low risk recall scores are 0.69 and 0.40.
  - The accuracy score plummets with undersampling. 0.544 is considerably low and would not be used. The precision again stays the same. The recall for high risk increases, while low risk decreases all the way down to 0.40. This is likely due to losing a lot of low risk data to undersample. This is not viable.
  
<img width="399" alt="image" src="https://user-images.githubusercontent.com/104074135/199359329-8ddca16f-3a36-43fb-9da7-8c1f31aa4aba.png">

<img width="496" alt="image" src="https://user-images.githubusercontent.com/104074135/199359351-b6b69a0c-a393-4ef7-815f-5de3a6287c2d.png">

- SMOTEENN
  -
  - The accuracy score for SMOTEENN is 0.648.
  - The high risk to low risk precision is 0.01 to 1.00.
  - The high risk and low risk recall scores are 0.72 and 0.57.
  - The accuracy score is between undersampling and SMOTE oversampling. This makes sense because it is a combination of the two. The precision stays the same as before. This is the best high risk recall seen at 0.72. The recall for both values is relative to naive oversampling +-0.01. This is very slightly better, but would not be used. 
  
<img width="366" alt="image" src="https://user-images.githubusercontent.com/104074135/199359429-f966af72-a13a-495b-89ff-3005d24dae0a.png">

<img width="489" alt="image" src="https://user-images.githubusercontent.com/104074135/199359472-31bcbc49-9b6a-4bd4-913b-bb345efb140f.png">

- Balanced Random Forest Classifier
  -
  - The accuracy score for balanced random forest classifier is 0.788.
  - The high risk to low risk precision is 0.04 to 1.00.
  - The high risk and low risk recall scores are 0.67 and 0.91.
  - The accuracy score here is 0.788 which is the best of all the tests so far. 0.788 is decent, but that still leaves 21% chance to incorrectly evaluate any customer. The recall is relatively average for high risk while low risk is the best so far. The precision for high risk also slightly increased.
  
<img width="421" alt="image" src="https://user-images.githubusercontent.com/104074135/199359551-25e3b6eb-e0cf-4afa-bcda-80bd9da01343.png">

<img width="491" alt="image" src="https://user-images.githubusercontent.com/104074135/199359574-aa51388d-e7ad-4f4b-a3eb-47c8b9d91a47.png">

- Easy Ensemble
  -
  - The accuracy score for easy ensemble is 0.925.
  - The high risk to low risk precision is 0.07 to 1.00.
  - The high risk and low risk recall scores are 0.91 and 0.94.
  - Easy ensemble has the highest score in every statistic. It drove high risk precision up to 0.07. Recall scores in above 0.9. Accuracy is at 0.925. Statstically easy ensemble is the best out of the six methods.
  
<img width="322" alt="image" src="https://user-images.githubusercontent.com/104074135/199359628-eabcd848-3ad9-4063-9b8f-1c37e7af59db.png">

<img width="496" alt="image" src="https://user-images.githubusercontent.com/104074135/199359644-55d567af-88ab-43a1-a76e-a7322a8a7059.png">

## Summary
To summarize, oversampling both naive and SMOTE are significantly better than undersampling. With undersampling we lose a lot of good data, and it just wouldn't work. SMOTEENN doesn't have the best scores, but compared to oversampling and undersampling I would chose this over the latter. The reason being is because it reduces bias while having similar statistics to naive oversampling. Next is balanced random forest classifier and while it has a higher accuracy, the high risk recall is too low to be a viable choice. The best out of the six is easy ensemble. The statistics blow everything else out of contention. Although, this could be a sign of excessive overfitting. In the end, I would not use any of the six methods. Nothing available has a high enough accuracy nor are the recall scores viable to accurately classify high and low risk. Even though easy ensemble looks the best statistically, it is more than likely due to excessive overfitting. 
