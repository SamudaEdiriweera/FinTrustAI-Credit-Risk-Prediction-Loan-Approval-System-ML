# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

#Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application's code into the container at /app
# This includes app.py, the src folder, and the model file
COPY . /app

# Expose port 8000 to allow communication with the app
EXPOSE 8000

# Define the command to run the application
# This runs the uvicorn server when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]