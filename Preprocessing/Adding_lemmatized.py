import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
 
# keywords (expandable) stored in txt file
with open("keywords.txt", 'r', encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]


lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in keywords]

print(lemmatized_words)

for word in lemmatized_words : 
    if word not in keywords :
        keywords.append(word)

with open("keywords.txt", "w") as f:
    f.write("\n".join(keywords))
