import pandas as pd
import string
import numpy as np
data = pd.read_csv("spam.csv",encoding="latin-1")
data = data.iloc[:,:2]
data.columns =["label","message"]
print(data.head())
def clean_text(text):
    text = text.lower()
    cleaned_text = "".join([char for char in text if char not in string.punctuation])
    return cleaned_text.split()
data['message']=data['message'].apply(clean_text)
print(data.head())
print(data.shape)
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

split = int(len(data_shuffled)*0.8)
train_data = data_shuffled.iloc[:split].reset_index(drop=True)
test_data = data_shuffled.iloc[split:].reset_index(drop= True)

print(f"Training set size:{len(train_data)}")
print(f"test set size: {len(test_data)}")
train_size = len(train_data)
spam_count = (train_data['label']=='spam').sum()
ham_count =(train_data['label']=='ham').sum()

# probability
p_spam = spam_count / train_size
p_ham = ham_count / train_size

print(f"p(spam):{p_spam}")
print(f"p(ham): {p_ham}")

spam_words_counts = {}
ham_words_counts = {}

for idx ,row  in train_data.iterrows():
    label = row['label']
    words = row['message']

    if label == 'spam':
            for word in words:
                spam_words_counts[word] = spam_words_counts.get(word, 0)+1
    else:
        for word in words:
            ham_words_counts[word]= ham_words_counts.get(word, 0) + 1 

print(f"unique words in spam: {len(spam_words_counts)}")
print(f"unique words in ham: {len(ham_words_counts)}")

n_spam = sum(spam_words_counts.values())
n_ham = sum(ham_words_counts.values())

vocabulary = set(list(spam_words_counts.keys()) + list(ham_words_counts.keys()))
n_vocabulary = len(vocabulary)

# ALpha for laplace smoothing
alpha = 1

print(f"total words in spam: {n_spam}")
print(f"total words in ham: {n_ham}")
print(f"Unique words (vocabulary size): {n_vocabulary}")

def classify(message):

    spam_score = np.log(p_spam)
    ham_score = np.log(p_ham)

    for word in message:

        spam_word_count = spam_words_counts.get(word, 0)+alpha
        p_word_given_spam = spam_word_count / (n_spam +alpha *n_vocabulary)
        spam_score += np.log(p_word_given_spam)

        ham_word_count = ham_words_counts.get(word, 0) + alpha
        p_word_given_ham = ham_word_count / (n_ham + alpha * n_vocabulary)
        ham_score += np.log(p_word_given_ham)
    
    if spam_score > ham_score:
        return 'spam'
    else:
        return 'ham'
    
sample_msg = test_data['message'].iloc[0]
print(f"Message: {sample_msg}")
print(f"Predicted: {classify(sample_msg)}")
print(f"Actual: {test_data['label'].iloc[0]}")

test_data['predicted'] = test_data['message'].apply(classify)
correct = (test_data['predicted'] == test_data['label']).sum()
total = len(test_data)
accuracy = (correct / total) * 100

print(f"Correctly Classified: {correct} / {total}")
print(f"Final Accuracy: {accuracy:.2f}%")


incorrect_cases = test_data[test_data['predicted'] != test_data['label']]
print("\n--- Examples of Misclassified Messages ---")
print(incorrect_cases[['message', 'label', 'predicted']].head())
