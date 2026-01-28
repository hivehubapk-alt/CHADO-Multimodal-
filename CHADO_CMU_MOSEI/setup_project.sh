#!/bin/bash

echo "Creating CHADO project structure..."

# Create all directories
mkdir -p src/data_processing
mkdir -p src/feature_extraction
mkdir -p src/models/{baseline,causal,hyperbolic,transport,mad,refinement,chado}
mkdir -p src/training
mkdir -p experiments/ablations
mkdir -p data/{raw,processed,splits}
mkdir -p checkpoints
mkdir -p results/{ablations,figures}
mkdir -p logs

# Create __init__.py files
touch src/__init__.py
touch src/data_processing/__init__.py
touch src/feature_extraction/__init__.py
touch src/models/__init__.py
touch src/models/baseline/__init__.py
touch src/models/causal/__init__.py
touch src/models/hyperbolic/__init__.py
touch src/models/transport/__init__.py
touch src/models/mad/__init__.py
touch src/models/refinement/__init__.py
touch src/models/chado/__init__.py
touch src/training/__init__.py
touch experiments/__init__.py
touch experiments/ablations/__init__.py

echo "✓ Project structure created!"
echo "Now you need to create the Python files."
echo "Use the create_files.py script next."

