# Use a lightweight Python base image
FROM python:3.11-bullseye

# Set the working directory inside the container
WORKDIR /app

# Create the log file and its directory
# This command ensures the file exists when the container starts

# Copy the requirements file and install dependencies
RUN apt update -y
RUN apt install s3fs -y
COPY ryn/ .
RUN echo "[global]" > /etc/pip.conf
RUN echo "index-url = https://mirror-pypi.runflare.com/simple/" >> /etc/pip.conf
RUN echo "trusted-host = mirror-pypi.runflare.com" >> /etc/pip.conf
RUN pip install -e .
RUN pip install clearml clearml-serving
RUN pip install uvicorn
RUN mkdir -p /mnt/s3
COPY ryn/api_test/mount_s3.sh /mnt/s3/
COPY clearml.conf /root
RUN cat /root/clearml.conf
WORKDIR api_test
# Copy the rest of the application code
COPY ryn/api_test/data_ingestion_test_api.py .

# Expose the port the API server will run on
EXPOSE 8169
RUN chmod -R 777 /var/log
RUN mkdir -p /var/log && touch /var/log/my-app.log
# Command to run the Uvicorn server when the container starts
CMD ["uvicorn", "data_ingestion_test_api:app", "--host", "0.0.0.0", "--port", "8169", "--no-access-log"]