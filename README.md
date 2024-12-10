# project-4

## Overview
The widespread dissemination of fake news on various platforms poses significant societal risks, leading to misinformation, societal conflicts, and loss of trust. The goal of this project is to develop a machine learning model in Python that can accurately classify news as either real or fake, aiding in the fight against misinformation.

## Hypothesis
Patterns in word usage, frequency, and structure differ significantly between fake and real news, enabling effective classification through text-based features.


## Dataset
The project used a publicly available [Fake News Detection Dataset](https://www.kaggle.com/datasets/subho117/fake-news-detection-using-machine-learning) containing labeled news articles.
Each record includes:
Text: The body of the news article.
Label: A binary label indicating whether the news is fake or real.


## Methodology
The methodology demonstrates a well-thought-out pipeline for fake news detection and prediction, utilizing a combination of Jupyter notebooks for data processing, analysis, and model training, and a Flask web application for real-time prediction. The pipeline integrates robust tools and frameworks, ensuring scalability and interpretability.

### Importing Libraries and Datasets
- The project employs a diverse set of Python libraries to equip the pipeline for various tasks:
  - **Data Manipulation:** Pandas for handling datasets across all files.
  - **Visualization:** Seaborn and Matplotlib for creating insightful visualizations.
  - **Machine Learning:** Scikit-learn for building and evaluating models.
  - **Text Processing:** NLTK for tokenization and stopword removal, and WordCloud for visual representation.
  - **Frameworks:** Flask for creating an interactive web application.
  - **Storage and Retrieval:** SQLite for managing data in a structured database format.
  - **Pretrained Model Integration:** Joblib in the Flask app for loading pre-trained models.
- The ETL process in **ETL.ipynb** cleans and prepares the dataset, storing it in an SQLite database. Subsequent notebooks (**Fake_News_Detection_using_Machine_Learning.ipynb** and **Fake_News_Optimized.ipynb**) retrieve this data for processing and analysis.

### Data Preprocessing
- Preprocessing steps are carefully designed to standardize and prepare the text data for analysis:
  - Handling missing values and duplicates ensures the dataset's integrity.
  - Text normalization techniques, such as special character removal, stopword elimination, and stemming/lemmatization, reduce noise and enhance model input quality.
  - In **Fake_News_Optimized.ipynb**, preprocessing includes advanced methods to ensure higher efficiency and better performance.

### Text Analysis and Visualization
- Exploration of word patterns between fake and real news articles provides valuable insights:
  - Word clouds and bar charts are used to visualize common terms, enhancing interpretability for both technical and non-technical stakeholders.
  - Distribution of fake vs. real news is analyzed to understand dataset characteristics.
  - Functions like "Get Top N Words" identify the most frequent terms, enriching the analysis.

### Converting Text into Vectors
- Numerical representation of text is achieved using robust methods:
  - **TF-IDF Vectorization:** Captures the importance of words relative to the dataset.
  - These methods provide structured input for machine learning models, facilitating effective classification.

### Model Training, Evaluation, and Prediction
- The project adopts a systematic approach:
  - Baseline models (Logistic Regression and Decision Tree) are trained and evaluated in **Fake_News_Detection_using_Machine_Learning.ipynb**.
  - Iterative optimization and hyperparameter tuning in **Fake_News_Optimized.ipynb** improve model performance.
  - Evaluation metrics, including accuracy, precision, recall, F1-score, and confusion matrices, ensure comprehensive assessment.
- In **Fake_News_Flask.py**, a pre-trained model and TF-IDF vectorizer are deployed for real-time prediction:
  - User input is vectorized, and predictions are rendered with detailed feedback on the web interface.
  - Counters track the proportions of fake and real news predictions dynamically.

### Flask Application
- **Fake_News_Flask.py** complements the machine learning pipeline with an interactive web application:
  - Users can input text directly into the interface.
  - Results, including prediction probabilities and classification as "Real News" or "Fake News," are displayed on the homepage.
  - The app also calculates and updates metrics for the percentages of fake and real news predictions.

## Expected Outcomes

### The deliverables include:

- A trained and tested model with high classification accuracy.
- Visualizations that shed light on fake news patterns.
- A reusable pipeline, providing value beyond this specific use case.
- These outcomes align with the stated objectives, promising a practical solution to fake news detection.

## Conclusion
Fake news articles have distinct linguistic patterns that can help machine learning models classify them, but these patterns often overlap with those in real news.
This captures the idea of leveraging linguistic patterns for classification while acknowledging the overlap observed in the word cloud analysis.
