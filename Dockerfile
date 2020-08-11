# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN apt update && apt install -y libgtk2.0-dev
RUN pip install Flask gunicorn flask-cors numpy opencv-python-headless cython
RUN python setup.py build_ext --inplace

# Start gunicorn server. 1CPU 8 threads
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
