# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from flask import Flask, render_template, request

import spacy
import pandas as pd

# Create the application object
app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output')
def tag_output():
#
       # Pull input
       some_input =request.args.get('user_input')

       # Case if empty
       if some_input == '':
           return render_template("index.html",
                                  my_input = some_input,
                                  my_form_result="Empty")
       else:
           some_image="giphy.gif"

           course = pd.read_csv('./data/courses_textrank_labels.csv')
           label_to_skill = dict(zip(course['label'], course['skill']))

           nlp = spacy.load('./models/train_textrank_labels')
           doc = nlp(some_input)
           some_output = []
           for ent in doc.ents:
               if label_to_skill.get(ent.text)==None:
                   some_output.append(ent.text)
               else:
                   some_output.append(label_to_skill.get(ent.text))


           form_result="NotEmpty"
           if not some_output:
               form_result="NotEmptyButNoContent"
           elif len(some_output[0])<=2:
               form_result="NotEmptyButNoContent"


           return render_template("index.html",
                              my_input=some_input,
                              my_output=some_output,
                              my_img_name=some_image,
                              my_form_result=form_result)


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/
