# Use a lightweight Python base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Create the log file and its directory
# This command ensures the file exists when the container starts

# Copy the requirements file and install dependencies
COPY ryn/ .
RUN echo "[global]" > /etc/pip.conf
RUN echo "index-url = https://mirror-pypi.runflare.com/simple/" >> /etc/pip.conf
RUN echo "trusted-host = mirror-pypi.runflare.com" >> /etc/pip.conf
RUN pip install -e .
RUN pip install clearml clearml-serving
RUN pip install uvicorn
ENV LAKEFS_ACCESS_KEY="AKIAJW35UUDJP54623UQ"
ENV LAKEFS_SECRET_KEY="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"
RUN mkdir -p /mnt/s3
COPY ryn/api_test/mount_s3.sh /mnt/s3/
COPY clearml.conf /root
WORKDIR api_test
# Copy the rest of the application code
# COPY log_reciver.py .
# COPY loop.sh .

# Expose the port the API server will run on
# EXPOSE 8787
# RUN chmod -R 777 /var/log
# RUN mkdir -p /var/log && touch /var/log/my-app.log
# Command to run the Uvicorn server when the container starts
# CMD ["uvicorn", "log_reciver:app", "--host", "0.0.0.0", "--port", "8787","--no-access-log"]
# RUN /bin/bash
