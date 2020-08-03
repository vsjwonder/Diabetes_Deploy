# from flask_table import Table, Col
# from wsgiref import simple_server
from flask import Flask, request, app
from flask import Response
from flask import render_template, request, jsonify
from flask_cors import CORS, cross_origin
import requests
import pandas as pd
from Diabetis_logistic_deploy import predObj

app = Flask(__name__, template_folder='templates', static_folder='assets')
CORS(app)
app.config['DEBUG'] = True


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


class ClientApi:

    def __init__(self):
        self.predObj = predObj()


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if request.method == 'POST':
            data = request.form.to_dict()
            print('data is:     ', data)
            #prediction
            pred = predObj()
            prediction1 = pred.predict_log(data)
            prediction = prediction1[0]
            print(prediction)
            #Statistics of model
            accuracy = prediction1[1]
            accuracy = accuracy * 100
            print(accuracy)
            f1 = prediction1[2]
            f1 = f1*100
            print(f1)
            precision = prediction1[3]
            precision = precision*100
            print(precision)
            recall = prediction1[4]
            recall = recall * 100
            print(recall)
            predmodel = prediction1[5]
            print(predmodel)
            #Table
            inputdata = pd.DataFrame({'Features': list(data.keys()),'Entered Values':list(data.values())})
            inputdata['Entered Values']=pd.to_numeric(inputdata['Entered Values'])
            print(inputdata)
            Nor = ["More No.= More Risk", "<100 mg/dl", "<80 mm Hg", 'M(12.5) F(18.5) mm', "1.6-16.6 mu U/ml",
                   "18.50 - 24.99 kg/m^2", '0.0 - 0.5', "High Age = High Risk"]
            inputdata['Normal Range'] = Nor
            print(inputdata)

            # inputdata = pd.DataFrame.from_dict(list(data.items()), )
            # print(inputdata)
            # inputdata = pd.Series(inputdata, name="Entered values")
            # inputdata.index.name="Features"
            # inputdata=inputdata.reset_index()

            # inputdata = inputdata.to_html()

            # print(inputdata)
            # inputdata=inputdata.transpose()
            # print(inputdata)

            # result = clintApp.predObj.predict_log(data)
            print('result is        ', prediction)
            return render_template('results.html', result=prediction, inputdata=inputdata, accuracy=accuracy,
                                   f1=f1, precision=precision, recall=recall, predmodel=predmodel)
        else:
            return render_template('results.html', result="value error")
    except Exception as e:
        print('exception is   ', e)
        # return Response(e)
        return render_template('index.html', result=e)


if __name__ == "__main__":
    clintApp = ClientApi()
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
    # httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    # httpd.serve_forever()
