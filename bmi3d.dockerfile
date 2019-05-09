FROM python:3

####### Set up directories and copy source code
RUN mkdir -v -p /code/src/

COPY . /code/bmi3d/
WORKDIR /code/bmi3d/

RUN mkdir -v -p /backup && chown root /backup
RUN mkdir -v -p /storage/plots && chown -R root /storage
RUN mkdir -v logs

# Prepare scripts: Fix line endings because windows breaks bash
RUN sed -i 's/\r$//' install/docker/package_install.sh
RUN sed -i 's/\r$//' install/docker/src_code_install.sh	

# Install required ubuntu packages
RUN ./install/docker/package_install.sh

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN ./install/docker/src_code_install.sh 
 
CMD [ "python", "./your-daemon-or-script.py" ]