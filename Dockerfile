# Use an official Python 3.8 image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update the package lists and install AWS CLI and Python dependencies
RUN apt-get update -y \
    && apt-get install -y awscli \
    && pip install --no-cache-dir -r requirements.txt

# Specify the command to run when the container starts
CMD ["python3", "app.py"]
