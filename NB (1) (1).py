#Author
# Lakshmi Praveena Rangavajhula
# Varun Suresh Parashar - vxp171830


# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
import re
import sys


#train_file_path = "/Users/varun/Data/UTD_Dairy/Spring2018/Machine_Learning/textTrainData.txt"
#test_file_path = "/Users/varun/Data/UTD_Dairy/Spring2018/Machine_Learning/textTrainData.txt"

train_file_path = sys.argv[1]
test_file_path = sys.argv[2]
# First we read in the dataset
train_data = pd.read_csv(train_file_path, sep='\t', lineterminator='\n',encoding='latin1')

#Add Headers to the file
headers = ["Sentence","Sentiment"]
train_data.columns = headers

#Print the Sample 3 records
print(train_data[0:3])


#Remove the Noisy record
train_data = train_data[~train_data[headers[1]].isnull()]

#Read the test data
test_data = pd.read_csv(test_file_path, sep='\t', lineterminator='\n',encoding='latin1')
test_data.columns= headers


# We need a function that will split the text based upon sentiment
def get_text(reviews, score):
  # Join together the text in the reviews for a particular sentiment.
  # We lowercase to avoid "Not" and "not" being seen as different words, for example.
   
    s = ""
    for index,row in reviews.iterrows():
        if row[headers[1]] == score:
            s = s +" "+ row['Sentence'].lower()
    s.replace("  "," ")
    return s

# Now we will capture the negative and positive samples in the training set.
# We will create two large strings, one of all text from positive reviews and one from the negatives
# We will then use these to create the word counts
# This will make the computations of the probabilities easier

# This will take a few minutes and use up some memory!

negative_train_text = get_text(train_data, 0)
positive_train_text = get_text(train_data, 1)

# We also need a function that will count word frequency for each sample
def count_text(text):
  # Split text into words based on whitespace.  Simple but effective.
  words = re.split("\s+", text)
  # Count up the occurence of each word.
  return Counter(words)


# Here we generate the word counts for each sentiment
negative_counts = count_text(negative_train_text)
# Generate word counts for positive tone.
positive_counts = count_text(positive_train_text)



# We need this function to calculate a count of a given classification
def get_y_count(score):
  # Compute the count of each classification occuring in the data.
  # return len([r for r in reviews if r[1] == str(score)])
    c = 0
    for index,row in train_data.iterrows():
        if row[headers[1]] == score:
            c = c + 1
    
    return c

# We need these counts to use for smoothing when computing the prediction.
positive_review_count = get_y_count(1)
negative_review_count = get_y_count(0)

# These are the class probabilities (we saw them in the formula as P(y)).
prob_positive = positive_review_count / len(train_data)
prob_negative = negative_review_count / len(train_data)

# Finallt, we create a function that will, given a text example, allow us to calculate the probability
# of a positive or negative review

def make_class_prediction(text, counts, class_prob, class_count):
  prediction = 1
  text_counts = Counter(re.split("\s+", text))
  for word in text_counts:
      # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
      # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
      # We also smooth the denominator counts to keep things even.
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Now we multiply by the probability of the class existing in the documents.
  return prediction * class_prob

# Here we will create a function that will actually make the prediction
def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
      return 0
    return 1

# Now we make predictions on the test data. Since it is a large set, we will simply select 200 movies.
predictions = [make_decision(row['Sentence'], make_class_prediction) for index,row in test_data.iterrows()]

#Reading the actual test Data
actual = test_data[headers[1]].tolist()

#Find the accuracy and print the accuracy
accuracy = sum(1 for i in range(len(predictions)) if predictions[i] == actual[i]) / float(len(predictions))
print("Accuracy")
print("{0:.4f}".format(accuracy))

#Print the Confusion matrix 
print(" Confusion matrix ")
print( confusion_matrix(test_data['Sentiment'], predictions))

