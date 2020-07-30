
#from flask_table import Table, Col
#from wsgiref import simple_server
from flask import Flask, request, app
from flask import Response
from flask import render_template, request,jsonify
from flask_cors import CORS,cross_origin
import requests
import pandas as pd
from Diabetis_logistic_deploy import predObj

app = Flask(__name__, template_folder='templates', static_folder='assets')
CORS(app)
app.config['DEBUG'] = True


@app.route('/',methods=['GET'])  # route to display the home page
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
            pred = predObj()
            prediction = pred.predict_log(data)

            inputdata=pd.DataFrame.from_dict([data])
            #inputdata=inputdata.transpose()

            #result = clintApp.predObj.predict_log(data)
            print('result is        ', prediction)
            return render_template('index.html', result=prediction, inputdata=inputdata)
        else:
            return render_template('index.html', result="value error")
    except Exception as e:
        print('exception is   ', e)
        #return Response(e)
        return render_template('index.html', result = e)


if __name__ == "__main__":
    clintApp = ClientApi()
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
    # httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    # httpd.serve_forever()
