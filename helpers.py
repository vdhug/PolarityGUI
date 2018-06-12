from flask import render_template
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score

def error(message, code=400):
    """Render message as an apology to user."""

    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("error.html", top=code, bottom=escape(message)), code


def getInfo(documents):
    info = documents.label.value_counts()
    return info


# Function that returns a dictionary with value equals to {Class: total_number_of_documents_of_the_class}
# For correct operation of the function, the variable 'documents' must had been read with
# 'pd.read_table(file1, header=None, names=['label', 'message'])'
def getDocumentsOfClasses(documents):
    classes = {}
    info = getInfo(documents)
    for i in range(0, len(info.keys())):
        name = info.keys()[i]
        number_of_documents = info[name]
        classes[name] = number_of_documents
    return classes


# Function that returns the classes existing in a set of documents
def getClasses(documents):
    classes = getDocumentsOfClasses(documents)
    return classes.keys()

# QUantidade de atributos apenas fazer  len(vect.get_feature_names())
# Depois de vetorizar a coleção e com ela transformada em matriz esparça pelo transform
# extrair numéro de documentos (x) e numero de tokens, atributos, "palavras" etc (y)
# Fazer  x, y = COLEÇÂO VETORIZADA.shape



def result(X, y, algoritmo, abordagem, metricas, particoes):
    abordagens = {"tf": False, "tp": True}
    nb = MultinomialNB()
    rl = LogisticRegression()
    algoritmos = {"nb": nb, "rl": rl}
    scoring = {"precisao": precision(), "revocacao": recall(), "f1": f1()}
    for metrica in metricas:
        resultado = {metrica: []}

    kf = StratifiedKFold(n_splits=particoes)
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]


        # import and instantiate CountVectorizer (with the default parameters)

        # instantiate the vectorizer
        vect = CountVectorizer(binary=abordagens[abordagem])
        # learn training data vocabulary, then use it to create a document-term matrix
        # The function fit creates an array with the words that appears in the collection
        vect.fit(X_train)
        X_train_dtm = vect.transform(X_train)
        # equivalently: combine fit and transform into a single step
        # X_train_dtm = vect.fit_transform(X_train)
        # transform testing data (using fitted vocabulary) into a document-term matrix
        X_test_dtm = vect.transform(X_test)
        # train the model using X_train_dtm (timing it with an IPython "magic command")
        algoritmos[algoritmo].fit(X_train_dtm, y_train)

        # make class predictions for X_test_dtm
        y_pred_class = algoritmos[algoritmo].predict(X_test_dtm)

        # calculate accuracy of class predictions
        from sklearn import metrics
        print(metrics.accuracy_score(y_test, y_pred_class))
        # print the confusion matrix
        print(metrics.confusion_matrix(y_test, y_pred_class))


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average=None)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)

