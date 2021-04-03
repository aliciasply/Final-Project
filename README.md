# “Fake News!” Detector
### https://fake-news-detector-project.herokuapp.com/


![Logo](Image/fake_news_logo.jpg)

## Background and Motivation: 

In the age of social media, it has become more important than ever to be wary of what we read on all platforms. Social media has made it easier to connect with the world and has great influence on the masses as anyone can share anything. What we wanted to do was to find a way to weed out the real news from the fake news. And, more importantly, we wanted to find a way to measure the impact that fake news has on society as a whole. In doing so, we created a machine learning model that predicts if articles are "real" and "fake".  With the model created, we ran through thousands of articles posted on Facebook from top news sources (BBC, Buzzfeed Politics, Conservative Post, CNN, Daily Mail, and Fox News).
Read our analysis here! https://docs.google.com/document/d/1i3ppTAVp6_3ZUh6Dt7e_Zl4LL0Ynhm-JyUPqZeueA9c/edit?usp=sharing

## Questions to Answer:

  * Which of these news sites contribute the most “Fake" news?
  * Which of these news sites contribute to the most "Real" news?
  * What are the top 5 most liked "Real"/"Fake" news articles?
  * What are the top 5 most commented "Real"/"Fake" news articles?
  * What are the top 5 most shared "Real"/"Fake" news articles?
  * What do the most engaged "Real"/"Fake" news articles have in common?
  * What are the most-used words in "Fake" news Vs. "Real" news?

## Technology

 * Machine Learning: sklearn (train_test_split, TfidfVectorizer, PassiveAggressiveClassifier, accuracy score, confusion_matrix, Pipeline, MultinomialNB)
 * Python/Pandas/Numpy
 * Flask (request/jsonify)
 * CORS
 * Newspaper (Article)
 * urllib
 * NLTK (punkt)
 * HTML/CSS/Bootstrap
 * Octoparse (webscraping tool)
 * Heroku

## Outline 

 * Create  machine learning model
 * Transform, Train & Test the Data
 * Analyze findings from Facebook Data
 * Visualizations 
 * Website Creation 
 * Create Flask app
 * Deployment to Heroku

## Data

* https://www.kaggle.com/hassanamin/textdb3 (training data)
* [BBC](https://www.bbc.com/)
* [BuzzFeed Politics](https://www.buzzfeednews.com/section/politics)
* [Conservative Post]()
* [CNN](https://www.cnn.com/)
* [Daily Mail](https://www.dailymail.co.uk/ushome/index.html)
* [Facebook](https://www.facebook.com/)
* [Fox News](https://www.foxnews.com/)

## Resources

*  https://newspaper.readthedocs.io/en/latest/
*  https://pypi.org/project/Flask-Cors/
*  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
*  https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
*  https://web.dev/cross-origin-resource-sharing/
*  https://www.kite.com/python/docs/nltk.punkt
*  https://getbootstrap.com/docs/4.0/components/forms/
*  https://getbootstrap.com/docs/4.0/components/buttons/
*  https://medium.datadriveninvestor.com/using-python-to-detect-fake-news-7895101aebb8
*  https://levelup.gitconnected.com/tweet-analysis-for-covid-19-fake-news-detection-bf7cbea5c12d
*  https://www.octoparse.com/signup?ref=homepage


### Data Analytics Team:
* [Miguel](https://github.com/52Godfrey) - Data collection via webscraping, Data cleaning, Flask creation and Heroku Deployment 
* [Paola](https://github.com/paola1395) - Data manipulation, Machine learning model creation, Flask creation, Logo Creator
* [Alicia](https://github.com/aliciasply) - Data cleaning, Data manipulation, Website creation and Visualizations
* [Medha](https://github.com/medha795) - Data cleaning, Data manipulation, Visualizations


