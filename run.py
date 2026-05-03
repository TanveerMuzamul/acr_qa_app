"""Run module for the MRI ACR QA application.

The comments in this file describe the main processing steps so the code is easier to review and maintain.
"""

from backend.app import app

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
