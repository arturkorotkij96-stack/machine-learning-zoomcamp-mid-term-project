# machine-learning-zoomcamp-mid-term-project
Reddit Comments Upvote Prediction

In this project, I aim to **predict upvoted comments on Reddit**.

For this, I selected a Kaggle dataset: https://www.kaggle.com/datasets/tiwarikaran/reddit-upvote-prediction?resource=download&select=Train_v2.csv

---

## Data Preparation and Feature Engineering

In the `data` folder, you can find **feature_engineering.py**, which generates extra features using the following packages:

* **BERT Model** (from Hugging Face Transformers): Produces 768-dimensional feature vectors from text.
    * *Documentation:* https://huggingface.co/docs/transformers/en/model_doc/bert
* **Latent Dirichlet Allocation (LDA) and CountVectorizer** (from scikit-learn): Used to discover topic probabilities (N Topics) for the free text input dataset.
    * *Documentation:* https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
* **Sentiment Score (VADER)**: Calculated using the NLTK VADER lexicon.
    * *Documentation:* https://www.nltk.org/api/nltk.sentiment.vader.html
* **Readability Score (Flesch-Kincaid Grade Level)**: Calculated using the `textstat` library.
    * *Documentation:* https://pypi.org/project/textstat/

**Note**: After all transformations, I reduced the BERT feature vector dimensions from **768** to **128** using **PCA transformation**. This was necessary because the full vector results were too resource-intensive for my laptop to process. Also, when I tried the 256-dimensional option, it **did not provide a tangible increase** in the AUC score.

---

## Model Tuning

In the `model` folder, you can find the `tune_model_parameters.py` script, which tunes parameters for four models:

* **Logistic Regression**: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
* **Random Forest Classifier**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
* **XGBoost Classifier**: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
* **SVM Classifier**: Included to **account for potential data non-linearity**.
    * *Documentation:* https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

I used **GridSearchCV** for parameter tuning and k-fold cross-validation. While this is a great utility, my laptop was not able to complete the full tuning cycles for the Random Forest Classifier. If you have more compute power, feel free to train it on the full dataset and check the results!

---

## Final Model Training

You can use `train_model.py` to train and save models with specific, non-tuned parameters. This script will save the **pickled models** to the root folder.

---

## Overall Results

I tested all of these models; it was a good learning experience that gave me a lot of practice. However, the results were **disappointing**, as the models only achieved an AUC score of about **0.66**, which is quite low.

Since GitHub does not allow file uploads greater than 100MB, I have only uploaded a compressed file, `processed_data_128_dim.csv.zip`, which contains 50k records. If you run `feature_engineering.py` on your local machine, it will produce the full dataset with 200k records (which should give better AUC results, likely around 0.65).

## 50k dataset AUC scores
```
model_lg_s auc: 0.5973414486736477
model_rf auc: 0.6240488891224658
model_xgb auc: 0.6054752926646608
svm model auc: 0.5704320731200665
```
## 200k dataset AUC scores [WIP]
```
For 200k dataset I recieved auc ~0.66 for model_xgb
```
## Conclusion

While I loved the engineering challenge, the initial models achieved a low **AUC score of ~0.66** (and ~0.60 on the 50k sample). This indicates the need for improvement, but is an invaluable learning point.

In the future, I believe I should be able to improve performance by:
* **Doing better data engineering** and extracting more features.
* **Trying to utilize the full 768-dimensional feature vector** from BERT.
* **Finding a better and/or bigger dataset.**
* **Using neural network models** to potentially achieve better results.