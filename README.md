# review-classification
A script in both Python and R to classify reviews with an NLP model from NLTK and tm.

# Why TSV instead of CSV?
Its likely that reviews could include commas, effectively ruining the way we read and write to CSV files. A TSV file means tab-separated, meaning that it uses the TAB as a delimiter over commas. This is a good choice, as its unlikely for us to see TABs in reviews.

# What is Bag of Words?
Its a method we use to extract all the words from the training set, and turning each of them into columns for use in the NLP model. Before we do this, its necessary to "clean the text" in order to get to a base text that makes something like 'love, lovely, and loving' have just one column.

# Fitting the models
For classification, we tested a Naive Bayes model against a Random Forest model, both in Python and R. We found that the Random Forest model attained a better accuracy over the Naive Bayes model, as the confusion matrix scored them 80% vs 73% respectively.

The dataset was obtained from superdatascience.com
