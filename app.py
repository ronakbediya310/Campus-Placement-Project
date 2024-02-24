from flask import Flask, render_template, request, jsonify
import os
from src.exception import CustomException
from src.utils import load_object
import pandas as pd
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData
from src.logger import logging
app = Flask(__name__)
pipeline = PredictionPipeline()

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('form.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                ssc_p=float(request.form.get('ssc_p')),
                ssc_b=request.form.get('ssc_b'),
                hsc_p=float(request.form.get('hsc_p')),
                hsc_b=request.form.get('hsc_b'),
                hsc_s=request.form.get('hsc_s'),
                degree_p=float(request.form.get('degree_p')),
                degree_t=request.form.get('degree_t'),
                workex=request.form.get('workex'),
                etest_p=float(request.form.get('etest_p')),
                specialisation=request.form.get('specialisation'),
                mba_p=float(request.form.get('mba_p'))
            )
            
            final_new_data = data.get_data_as_dataframe()
            prediction_pipeline = PredictionPipeline()
            logging.info(final_new_data)
            pred = prediction_pipeline.predict(final_new_data)
            result = round(pred[0], 2)
            if result == 1:
                conclusion="Congratulations! You will be placed All the best"
            else :
                conclusion="need improvement on Your Skills  All the best"
            return render_template('form.html', final_result=conclusion)
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
