from .visualization.dash_app import create_dash_app

def main():
    app = create_dash_app()
    app.run_server(debug=True)

if __name__ == "__main__":
    main()