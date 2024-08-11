# Amazon Rating Predictor
An application that predicts Amazon ratings from text reviews using a Multinomial Naive Bayes classifier. Based on an Amazon product page URL provided, reviews are scraped from the page and then classified into {1, 2, 3, 4, 5} star ratings.

# Handling Unbalanced Data
![](https://github.com/Amber-Abuah/Amazon-Rating-Predictor/blob/main/RatingDistribution.jpg)  
The dataset had heavily imbalanced data, with a very large majority of them being 5 star reviews. Because of this the model initially predicted all reviews as 5 stars, no matter the text review. To fix this, SMOTEENN (Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors) was used which generated synthetic samples for underepresented classes and removed samples that could not be predicted by KNN. After applying this technique the model's accuracy increased to 91.46%.

Gradio Depolyment: https://huggingface.co/spaces/sweetfelinity/AmazonRatingPredictor

# Libraries Used
BeatifulSoup, NLTK, Gradio, Scikit-Learn, Pandas, Imblearn

amazon_reviews.csv from 
