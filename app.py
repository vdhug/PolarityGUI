import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/login')
def login():
    return render_template("login.html")


# Rota index, seleção e leitura da base de dados.
@app.route('/file', methods=["POST"])
def file():
    file1 = request.files["file1"]

    documents = pd.read_table(file1, header=None, names=['label', 'message'])
    print(documents.shape)
    print(documents.head(10))
    print(documents.label.value_counts())

    X = documents.message
    y = documents.label

    return render_template("test.html")


if __name__ == '__main__':
    app.run()
