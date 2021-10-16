from wsgiref import simple_server #The Web Server Gateway Interface (WSGI) is a standard interface between web server software and web applications written in Python.
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS,cross_origin
#from prediction_validation_insertion import pred_validation
from TrainingModel import train_model
from training_validation_insertion import train_validation
import flask_monitoringdashboard as dashboard
#from predict_from_model import prediction
import joblib
import numpy as np


#A correctly configured terminal session has the LANG environment variable set; it describes which encoding the terminal expects as output from programs running in this session.
os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')

app=Flask(__name__)
#dashboard.bind(app)
#CORS(app)
@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route("/predict",methods=['POST','GET'])
@cross_origin()
def index():
    if request.method == 'POST':

        try:
            Age=float(request.form.get('Age',False))
            print(Age)

            Sex= (request.form.get('Sex',False))
            if Sex=="Female":
                Sex=0
            elif Sex=="Male":
                Sex=1

            On_thyroxine = (request.form.get('on_thyroxine', False))
            if On_thyroxine == "False":
                On_thyroxine = 0
            elif On_thyroxine == "True":
                On_thyroxine = 1

            query_on_thyroxine = (request.form.get('query_on_thyroxine', False))
            if query_on_thyroxine == "False":
                query_on_thyroxine = 0
            elif query_on_thyroxine == "True":
                query_on_thyroxine = 1

            On_antithyroid_medication = (request.form.get('on_antithyroid_medication', False))
            if On_antithyroid_medication == "False":
                On_antithyroid_medication = 0
            elif On_antithyroid_medication == "True":
                On_antithyroid_medication = 1

            sick = (request.form.get('sick', False))
            if sick == "False":
                sick = 0
            elif sick == "True":
                sick = 1

            pregnant = (request.form.get('pregnant', False))
            if pregnant == "False":
                pregnant = 0
            elif pregnant == "True":
                pregnant = 1

            thyroid_surgery = (request.form.get('thyroid_surgery', False))
            if thyroid_surgery == "False":
                thyroid_surgery = 0
            elif thyroid_surgery == "True":
                thyroid_surgery = 1



            I131_treatment = (request.form.get('I131_treatment', False))
            if I131_treatment == "False":
                I131_treatment = 0
            elif I131_treatment == "True":
                I131_treatment = 1

            query_hypothyroid = (request.form.get('query_hypothyroid', False))
            if query_hypothyroid == "False":
                query_hypothyroid = 0
            elif query_hypothyroid == "True":
                query_hypothyroid = 1

            query_hyperthyroid = (request.form.get('query_hyperthyroid', False))
            if query_hyperthyroid == "False":
                query_hyperthyroid = 0
            elif query_hyperthyroid == "True":
                query_hyperthyroid = 1

            lithium = (request.form.get('lithium', False))
            if lithium == "False":
                lithium = 0
            elif lithium == "True":
                lithium = 1

            Goitre = request.form.get('goitre', False)
            if Goitre == "False":
                Goitre = 0
            elif Goitre == "True":
                Goitre = 1

            tumor = request.form.get('tumor', False)
            if tumor == "False":
                tumor = 0
            elif tumor == "True":
                tumor = 1

            Hypopituitary = (request.form.get('hypopituitary', False))
            if Hypopituitary == "False":
                Hypopituitary = 0
            elif Hypopituitary == "True":
                Hypopituitary = 1


            psych = (request.form.get('psych', False))
            if psych == "False":
                psych = 0
            elif psych == "True":
                psych = 1

            T3 = (request.form.get('T3', False))
            if T3 == "False":
                T3 = 0
            elif T3 == "True":
                T3 = 1
            Total_thyroxine_TT4 = float(request.form.get('TT4', False))


            T4U = float(request.form.get('T4U', False))
            FTI = float(request.form.get('FTI', False))

            referral_source = (request.form.get('referral_source', False))
            if referral_source == "SVI":
                referral_source = 1
            elif referral_source == "other":
                referral_source = 1
            elif referral_source == "SVHD":
                referral_source = 1
            elif referral_source == "STMW":
                referral_source = 1















        # values=({"age":Age,"sex":Sex,"TSH":Level_thyroid_stimulating_hormone,
        #         "FTI":Free_thyroxine_index,"on_thyroxine":On_thyroxine,
        #         "on_antithyroid_medication":On_antithyroid_medication,
        #         "goitre":Goitre,"hypopituitary":Hypopituitary,
        #         "psych":Psychological_symptoms,"T3_measured":T3_measured})
        # my_data=db.insert_one(values)
            model=joblib.load('KNN.pkl')
            print("model done")


            arr=np.array([[Age,Sex,On_thyroxine,query_on_thyroxine,On_antithyroid_medication,
            sick,pregnant,thyroid_surgery,I131_treatment,query_hypothyroid,query_hyperthyroid,lithium,Goitre,tumor,Hypopituitary,psych,T3,Total_thyroxine_TT4,T4U,FTI,referral_source,0,0,0,0]])
            print(arr)
            pred=model.predict(arr)
            pred=pred[0]
            if pred==0:
                pred="Primary Thyroid"
                return render_template('results.html', prediction=pred)
            elif pred==2:
                pred="compensated_hypothyroid"
                return render_template('results.html', prediction=pred)
            else:
                pred="Negative"
                return render_template('results.html', prediction=pred)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json['folderpath'] is not None:
            path = request.json['folderpath']

            train_val_obj = train_validation(path)
            train_val_obj.train_validation()

            train_model_obj = train_model()
            train_model_obj.trainingModel()

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)

    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training Successfull!!!")

#port = int(os.getenv("PORT"))
if __name__ == '__main__':
    app.run(debug=True)