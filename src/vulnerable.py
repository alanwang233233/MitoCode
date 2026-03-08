"""
Intentionally vulnerable code to verify CodeQL security scanning is operational.
DO NOT use any of these patterns in production code.
"""
import os
import sqlite3
import subprocess
import pickle
import flask


app = flask.Flask(__name__)


# CWE-89: SQL Injection
# CodeQL rule: py/sql-injection
def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchall()


# CWE-78: OS Command Injection
# CodeQL rule: py/command-line-injection
def run_command(user_input):
    return subprocess.check_output("echo " + user_input, shell=True)


# CWE-22: Path Traversal
# CodeQL rule: py/path-injection
def read_file(filename):
    base_dir = "/var/data"
    path = os.path.join(base_dir, filename)
    with open(path) as f:
        return f.read()


# CWE-502: Deserialization of Untrusted Data
# CodeQL rule: py/unsafe-deserialization
def load_data(data: bytes):
    return pickle.loads(data)


# CWE-601: Open Redirect
# CodeQL rule: py/url-redirection
@app.route("/redirect")
def redirect_user():
    url = flask.request.args.get("url", "/")
    return flask.redirect(url)


# CWE-79: Cross-Site Scripting (Reflected XSS)
# CodeQL rule: py/reflected-xss
@app.route("/greet")
def greet():
    name = flask.request.args.get("name", "")
    return flask.make_response("<h1>Hello, " + name + "!</h1>")
