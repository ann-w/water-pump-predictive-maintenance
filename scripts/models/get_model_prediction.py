import pickle


def load_model(filename: str):
    model = pickle.load(open(filename, 'rb'))
    return model


def make_prediction(model, observation):
    """Takes as input a trained sklearn model to make a prediction"""
    prediction = model.predict(observation)
    return prediction
