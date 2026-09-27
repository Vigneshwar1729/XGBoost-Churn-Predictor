# Use the official Python slim image for a smaller footprint
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install PyTorch CPU first to save massive amounts of space/memory on Render
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file and install the rest of dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and BOTH models
COPY app.py .
COPY xgboost_churn_model.json .
COPY ft_transformer_model/ ./ft_transformer_model/

# Expose the port Uvicorn runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
