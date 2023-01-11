from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import pickle



app=Flask(__name__)
model3=pickle.load(open('model3.pkl','rb'))



@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        

        Location = request.form['Location']

        windgustdir = int(request.form['windgustdir'])

        winddir9am = int(request.form['winddir9am'])

        winddir3pm = int(request.form['winddir3pm'])

        
        maxtemp = float(request.form['maxtemp'])

        

        windgustspeed = float(request.form['windgustspeed'])

        

        humidity3pm = float(request.form['humidity3pm'])

        

        pressure3pm = float(request.form['pressure3pm'])

        
        raintoday = int(request.form['raintoday'])

        # storing the data in 2-D array
        predict_list = [[Location,  maxtemp, 
                        windgustdir, windgustspeed, winddir9am, winddir3pm, 
                          humidity3pm,  pressure3pm,
                         raintoday, ]]

        

        # predicting the results using the model loaded from a pickle file(logreg.pkl)
        output = model3.predict(predict_list)

        # loading the templates for respective outputs(0 or 1)
        if output == 0:
            return render_template("sunnyday.html")
        else:
            return render_template("rainyday.html")

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)