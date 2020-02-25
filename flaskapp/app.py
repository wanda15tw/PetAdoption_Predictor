from flask import Flask, render_template, request
import pandas as pd
import pickle, os

import sys
sys.path.insert(1, '/Users/wandajuan/Documents/2020/08 PetAdoption')
from src.d01_data.read_data import *
from src.d02_intermediate.preprocessing import *
from src.d03_processing.build_features import *
from src.d04_modeling.train_model import *
from src.d04_modeling.predict_model import *

HOME = '/Users/wandajuan/Documents/2020/08 PetAdoption'
MODEL_DIR = 'data/04_models'
path = os.path.join(HOME, MODEL_DIR)

app = Flask(__name__)



strings = {
    "Type": ['Cat', 'Dog'],
    "top_breed":['Tabby', 'Domestic Medium Hair', 'Mixed Breed', 'Domestic Short Hair',
				 'Domestic Long Hair', 'Terrier', 'rare', 'Persian', 'Rottweiler', 'Shih Tzu',
				 'Siamese', 'American Shorthair', 'Spitz', 'Labrador Retriever',
				 'Golden Retriever', 'German Shepherd Dog', 'Calico', 'Beagle',
				 'Oriental Short Hair', 'Poodle', 'Bengal'],
    "Gender": ['Male', 'Female', 'Other'],
    "Color1": ['Black', 'Brown', 'Cream', 'Gray', 'Golden', 'White', 'Yellow'],
    "MaturitySize": ['Small', 'Medium', 'Large', 'Extra Large'],
    "FurLength": ['Short', 'Medium', 'Long'],
    "Vaccinated": ['No', 'Not Sure', 'Yes'],
    "Dewormed": ['No', 'Not Sure', 'Yes'],
    "Sterilized": ['No', 'Not Sure', 'Yes'],
    "Health": ['Healthy', 'Minor Injury', 'Serious Injury'],
    "State": ['Selangor', 'Kuala Lumpur', 'Perak', 'Pulau Pinang', 'Terengganu', 'Johor',
			 'Melaka', 'Pahang', 'Kedah', 'Negeri Sembilan', 'Sabah', 'Sarawak', 'Kelantan',
			 'Labuan'],
    "top entity": ['rare', 'mother', 'guard dog', 'boy', 'cat', 'anyone', 'dog', 'nan', 'owner',
					 'adoption', 'one', 'home', 'breed', 'siblings', 'friend', 'girl', 'someone',
					 'adopters', 'baby', 'adopter', 'pet', 'female', 'family', 'companion', 'name',
					 'all', 'people', 'house', 'everyone', 'sisters', 'babies']
}

# min, max, default value
floats = {
    "docsentiment-score": [-1, 1, 0]
}

# min, max, default value
ints = {
    "Age": [0, 256, 10],
    "breed_cnt": [0, 100, 2],
    "color_cnt": [1, 3, 1],
    "Quantity": [1, 20, 1],
    "Fee": [0, 3000, 0],
    "VideoAmt": [0, 10, 0],
    "PhotoAmt": [0, 30, 3],
    "lda_topic": [0, 9, 0],
    "nmf_output": [0, 9, 0],
    "Desc_WC": [0, 1250, 60]
}

# {0: 'Same Day', 1: 'One Week', 2:'One Month', 3:'2 - 3 Month', 4:'No Adoption'}
labels = ["Same Day", "One Week", "One Month", "2 - 3 Month", "No Adoption"]

def generate_input_lines():
    result = f'<table>'

    counter = 0
    for k in strings.keys():
        if (counter % 2 == 0):
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<select class="form-control" name="{k}">'
        for value in strings[k]:
            result += f'<option value="{value}" selected>{value}</option>'
        result += f'</select>'
        result += f'</td>'
        if (counter % 2 == 1):
            result += f'</tr>'
        counter = counter + 1

    counter = 0
    for k in ints.keys():
        minn, maxx, vall = ints[k]
        if (counter % 2 == 0):
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<input type="number" class="form-control" min="{minn}" max="{maxx}" step="1" name="{k}" id="{k}" value="{vall}" required (this.value)">'
        result += f'</td>'
        if (counter % 2 == 1):
            result += f'</tr>'
        counter = counter + 1

    counter = 0
    for k in floats.keys():
        minn, maxx, vall = floats[k]
        if (counter % 2 == 0):
            result += f'<tr>'
        result += f'<td>{k}'
        result += f'<input type="number" class="form-control" min="{minn}" max="{maxx}" step="1" name="{k}" id="{k}" value="{vall}" required (this.value)">'
        result += f'</td>'
        if (counter % 2 == 1):
            result += f'</tr>'
        counter = counter + 1

    result += f'</table>'

    return result


app.jinja_env.globals.update(generate_input_lines=generate_input_lines)


def processing_w_train(input):
	X_train = pd.read_hdf('/Users/wandajuan/Documents/2020/08 PetAdoption/data/02_intermediate/intermediate3.h5', key='train')
	df = X_train.append(input)

	# Processing to build features
	## feature selection
	df = df.loc[:, ['Type', 'Age', 'top_breed', 'breed_cnt', 'Gender', 'Color1', 'color_cnt', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
	   'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt', 'docsentiment-score', 'top entity', 
	   'lda_topic', 'nmf_output', 'Desc_WC']]

	## Imputation for missing values
	from sklearn.impute import SimpleImputer
	imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	df['top entity'] = imputer.fit_transform(df['top entity'].values.reshape(-1,1))

	df.loc[df['top entity'].isnull()]
	df.loc[df['docsentiment-score'].isnull(), 'docsentiment-score'] = 0
	

	## Encoding
	df = encoding(df)
	## Scaling for numerical features
	df = scaling(df)

	return df

def predict(X):
	model = pickle.load(open(os.path.join(path, 'best_model_rf'), 'rb'))
	return model.predict(X)


@app.route('/', methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		input = {}
		for k, v in request.form.items():
			try:
				input[k] = float(v)
			except:
				input[k] = v

		input = pd.DataFrame(input, index=[99999])
		X = processing_w_train(input)
		y = predict(X)
		prediction = labels[y[-1]]

		return render_template('predict.html', input=input, predict=prediction)
	else:
		return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True)