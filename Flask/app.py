from flask import Flask, render_template, request
from blink_count import eye_blink


app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def index():  # put application's code here
    if request.method == 'POST':
        eye_blink()
        return render_template('index.html')

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
