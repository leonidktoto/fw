# Use the official Python image as a base
FROM python:3.11.9-slim


RUN apt-get update && apt-get install -y ffmpeg && apt-get clean
# Set the working directory inside the container
WORKDIR /app

# Install the dependencies
RUN pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
RUN pip install --force-reinstall ctranslate2==3.24.0

# Copy the rest of the application code to the container
COPY . .

# Set the command to run your script
CMD ["python", "main.py"]
