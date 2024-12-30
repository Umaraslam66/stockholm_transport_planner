# setup_project.py
import os
import shutil

def create_project_structure():
    # Project root directory
    root_dir = "stockholm_transport_planner"
    
    # Define the directory structure
    directories = [
        "",  # Root directory
        "src",
        "src/models",
        "src/utils",
        "src/data",
        "src/visualization",
        "tests",
        "docs",
        "notebooks",
        "data"
    ]
    
    # Create directories
    for dir_path in directories:
        full_path = os.path.join(root_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        
    # Create necessary files
    files = {
        "README.md": """# Stockholm Transport Planner

A multimodal transportation connection planner using machine learning for route optimization.

## Features
- Multi-modal route planning (train, metro, bus)
- ML-based delay prediction
- Interactive visualization
- Weather impact analysis

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python -m src.main
```

## Project Structure
- `src/`: Source code
  - `models/`: Core data models and ML components
  - `utils/`: Utility functions
  - `data/`: Data handling
  - `visualization/`: Plotting and Dash app
- `tests/`: Unit tests
- `docs/`: Documentation
- `notebooks/`: Jupyter notebooks
- `data/`: Sample data and resources
""",
        
        "requirements.txt": """networkx>=2.5
pandas>=1.2.0
numpy>=1.19.0
plotly>=4.14.0
dash>=2.0.0
scikit-learn>=0.24.0
python-dotenv>=0.19.0
""",
        
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# VS Code
.vscode/

# Environment
.env
.venv
venv/
ENV/

# Project specific
*.log
data/*.csv
""",
        
        "src/__init__.py": "",
        "src/models/__init__.py": "",
        "src/utils/__init__.py": "",
        "src/data/__init__.py": "",
        "src/visualization/__init__.py": "",
    }
    
    for file_path, content in files.items():
        with open(os.path.join(root_dir, file_path), 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")