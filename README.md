# Skill Me - Extract skills from course descriptions
Skill Me is a web app that extracts key job skills from text of course titles or descriptions. 

You can view the slides I made that describe the project here : [SkillMe Demo](https://github.com/mollyteng/insight/blob/master/SkillMe_Demo.pdf)

## Summary
Skill Me is a web app that uses Named Entity Recognizion (NER) in NLP to identify predifined key job skills from text of course titles or descriptions.

[SkillMe Development](https://github.com/mollyteng/insight/tree/master/skill_me) contains the development steps of SkillMe.

[SkillMe Application](https://github.com/mollyteng/insight/tree/master/webapp) contains the source code for the web app.

## Context
Survey shows that 40% of employers can not find the talents with the right skills they need, while for individuals, it is hard to find personalized career and learning pathways to achieve their goals. It would be great if we can build an AI-powered platform that matches people, business and learning resources. As a first step, I developped SKill Me, an NLP-based model that automates the process of extracting job skills from various sources of texts.

This was a consulting project I worked on for FutureFit AI as an Insight Data Science Fellow.

## Data
I got 3164 courses with titles and descriptions from FutureFit AI's database in semi-structured json format. 853 of the courses were manually labeled with skill tags. Among them, there were 224 unique skill tags. 
 
I used BeautifulSoup, nltk, and string libraries to preprocess the texts. I kept course descriptions in English only, removed HTML tags, stop words and punctuations, and lower-cased, tokenized and lemmatized the words.

## Training Named Entity Recognizer
I used Named Entity Recognition to solve the problem. I trained a new entity called "job skills" using spaCy library by starting from an empty model. I passed the training instances to the model in two ways:

- For courses that had exact wording appearing in the text (title+description), I passed the skill tags with their corresponding indices directly to the model;
- For courses that didn't have exact wording appearing anywhere in the text (title+description), I used Gensim TextRank to generate keywords, trained the model on the keywords, and then mapped the keywords back to the skill tags.

## Validation
I calculated semantic distance between the model-predicted job skills and the true skill tags using cosine similarity based on pretrained Word2Vec embeddings (enWiki + News). I obtained the following accuracies:

- Training mean cosine similarity: .90
- Test mean cosine similarity: .73

## Web Application
I built a simple web application to demonstrate the model using Python and Flask. It was deployed on Google Cloud Platform, accessiable at https://skillfinder.appspot.com/

Here are example results of the web app:
Input a course description:
![Webapp Results1](webapp_results1.png)
Return skills identified:
![Webapp Results2](webapp_results2.png)
When no predefined skills identified:
![Webapp Results3](webapp_results3.png)
