from client import create_app

# Create the Flask app
application = app = create_app()

if __name__ == "__main__":
    application.run(threaded=True, debug=True)