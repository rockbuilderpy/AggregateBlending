import flask
from flask import Flask, request
from flask_cors import CORS, cross_origin
import process

app = Flask(__name__)
CORS(app)


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    print(data)
    status = process.calculate(data['input2'], data['input3'])
    return status


if __name__ == '__main__':
    app.run(debug=True)