# Use a base image with Python and necessary dependencies
FROM python:3.9-slim

# Set a working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Set up environment variables if needed
ENV PYTHONPATH "${PYTHONPATH}:/app/src"

# Command to run the main script
CMD ["python", "src/main.py"]
