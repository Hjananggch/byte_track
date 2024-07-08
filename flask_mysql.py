from flask import Flask, request, jsonify, render_template, redirect
import MySQLdb

app = Flask(__name__)
db = MySQLdb.connect(host="localhost", port='3306', user="root", passwd="password", db="video_save")

@app.route('/')
def index():
    cursor = db.cursor()
    cursor.execute("SELECT * FROM video_save")
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        file = file.filename
        content = file.filename
        cursor = db.cursor()
        cursor.execute("INSERT INTO video_save (filename, size, format, content) VALUES (%s, %s,%s,%s)", (filename, size, format, content))
        db.commit()
        return redirect('/')
