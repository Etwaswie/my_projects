This repository was created to describe my completed projects, as well as projects currently in progress.

# Books recommended
**Description:** This is my latest project, which is currently under development. It aims to create a recommendation service that offers similar books based on the user's indication of the book they like. An integral part of this project is the "Book scraping project", which includes a web scraper designed to extract comprehensive book information from the [book24.ru](https://book24.ru/?ysclid=m2343foj7s932659211) website. The scraper gathers essential data such as ISBN, titles, authors, genres, publication year, age restrictions, number of pages, publishers, ratings, and descriptions to enhance the recommendation engine.

**Technologies:** Clickhouse, requests, selenium, NLP, Recsys.

**Links:** https://github.com/Etwaswie/books-recommender, https://github.com/Etwaswie/books_scraping

--------------------------------------------------------------------

# ML Service with billing system
**Description:** This project is an ML service with a billing system. The ultimate goal of the service is:

- For administrators, the ability to upload models for a predicate based on user test data, set the cost of one prediction of a specific model.
- For users, the ability to buy a model, select a model from a list of all available ones, upload data to it and receive predictions at the output.

Currently, the frontend part of the project is being finalized for more comfortable use.

**Technologies:** FastAPI, uvicorn, React, Bootstrap, SQLite.

**Link:** https://github.com/Etwaswie/ML-service

----------------------------------------------------------------

# Recommendation system for KION cinema database
**Description:** This project involved developing a microservice for generating real-time recommendations. The system integrates multiple recommendation models, ranging from basic approaches such as popularity and random selections to more advanced algorithms, including kNN (both item-based and user-based), LightFM, DSSM, and MultiVAE.

**Technologies:** kNN (item kNN, user kNN), LightFM, DSSM, MultiVAE, LightGBM, optuna, RecTools, implicit, RecBole.

**Link:** https://github.com/Etwaswie/RecommendedSystemService

----------------------------------------------------------------

# Geo-Platform service for analyzing and visualizing retail point ratings
**Description:** Developed as part of a challenge for X5 Group during a hackathon, this project involved parsing reviews of retail points from publicly available sources, analyzing sentiment using NLP tools, and classifying data into groups. Each store receives ratings based on various criteria and is displayed on a map with markers.

**Technologies:** Python, Pandas, BeautifulSoup, Requests, Transformers, Streamlit.

**Link:** https://github.com/Etwaswie/ai_talent_hackathon

**Service:** https://geoplatforma.streamlit.app

-----------------------------------------------------------------

# Traffic sign recognition assistant
**Description:** The project involves the development of a prototype assistant for drivers that alerts them to traffic signs. The model utilizes the mAP 0.5 metric to evaluate detection accuracy, aiming for frequent recognition of signs. Initial results showed a mAP of 0.24, attributed to the complexity of similar signs. Experiments were conducted using YOLOv8n with varying classes and hyperparameters, resulting in significant improvements in performance. The prototype is optimized for deployment on mobile devices, specifically using NCNN for enhanced inference speed.

**Technologies:** YOLOv8n, PyTorch, NCNN, Raspberry Pi 4B, OpenVINO, ORT, RKNN.

**Link:** https://github.com/Etwaswie/DeepLearning

-----------------------------------------------------------------

# Resume-job matching system
**Description:** The project aims to develop a software solution for matching job descriptions with resumes. It utilizes a dataset of users and resumes sourced from hh.ru through custom web scraping. The core of the solution is based on the sentence-transformer algorithm, which facilitates effective matching. A user-friendly interface displays recommendations, providing three job suggestions for each of the 9,923 users. Various models, including Doc2Vec and FastText, were evaluated and optimized for performance, resulting in efficient and relevant recommendations.

**Technologies:** Sentence-Transformer, Doc2Vec, FastText, Streamlit.

**Link:** https://github.com/Etwaswie/DeepLearningPartTwo

-----------------------------------------------------------------

# Predicting Answer Ratings on StackOverflow with E5-small-v2 Model
**Description:** This project aims to improve the quality of answers on StackOverflow. The goal was to develop a helper tool that integrates into the forum interface and evaluates answers in real-time as users type.

**Technologies:** Transformers, Scikit-learn, TF-IDF, XGBoost, NLP.

**Link:** https://github.com/Etwaswie/StackOverflow

------------------------------------------------------------------

# NLP course
**Description:** I completed a training course in NLP, during which I worked on the following tasks:

- Spam Classification: Utilizing FastText, Word2Vec, convolutional and recurrent neural networks.
- Topic Modeling: Using BigARTM and LDA.
- Text Classification: Implementing distilbert-base-uncased, a version of the BERT model.
- Question Answering: Applying RuBERT.

**Technologies:** 
- Libraries: Corus, NLTK, spaCy, scikit-learn, gensim, Transformers, pyLDAvis.
- Models and Techniques: CountVectorizer, TfidfVectorizer, DecisionTreeClassifier, LogisticRegression, Naive Bayes, Word2Vec, fastText, CNN, LSTM, LDA, BigARTM

**Link:** https://github.com/Etwaswie/NLP-Course
