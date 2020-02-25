from flask import Flask, render_template, request

app = Flask(__name__)


tasks = [1, 2, 3, 4, 5]


@app.route('/', methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		task_content = request.form['content']
		return task_content
	else:
		pass
	return render_template('index.html')


	## read inputs, determine if all valid
	## preprocessing, build features
	## load model and generate predictions
	## redirect to display predictions

if __name__ == "__main__":
	app.run(debug=True)