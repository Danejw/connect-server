# Use slim Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app files
COPY . .

# Set environment variables (optional if using .env)
ENV PYTHONUNBUFFERED=1

# Run the MCP server
CMD ["python", "main.py"]
