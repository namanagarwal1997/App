from flask import Flask,render_template,url_for,request,jsonify
from sklearn.externals import joblib
from flask_material import Material

from sklearn import metrics

app = Flask(__name__)
Material(app)


#Loading the model
MODEL = joblib.load("iris.pkl")
MODEL_LABELS = ["setosa", "veriscolor", "virginica"]


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    data=pd.read_csv("data/Iris.csv")
    return render_template("preview.html",df_view=data)


@app.route('/', methods=['POST'])
def analyze():
	# Retreive query parameters related to this request.
	sepal_length = request.form["sepal_length"]
	sepal_width = request.form["sepal_width"]
	petal_length = request.form["petal_length"]
	petal_width = request.form["petal_width"]
	model_choice=request.form['model_choice']

	features = [[sepal_length, sepal_width, petal_length, petal_width]]
	
	# Use the model to predict the classes
	label_index = MODEL.predict(features)
	# Retrieve the iris name that is associated with the predicted class
	label = MODEL_LABELS[label_index[0]]
	# Create and send a response to the API caller
	# return jsonify(status="complete", label="label")
	# return render_template("index.html")
	return render_template('index.html',petal_width=petal_width,sepal_width=sepal_width,sepal_length=sepal_length,petal_length=petal_length,result=label,model_choice=model_choice)
if __name__ == "__main__":
	app.run(debug=True)