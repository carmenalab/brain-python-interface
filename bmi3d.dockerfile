FROM python:3

RUN echo $HOME

RUN apt-get install apt

RUN mkdir $HOME/code/
COPY . $HOME/code/bmi3d/
WORKDIR $HOME/code/bmi3d/

RUN echo $HOME
RUN echo $PWD

RUN sed -i 's/\r$//' install/docker_install.sh	# Fix line endings because windows breaks bash
RUN ./install/docker_install.sh

CMD [ "python", "./your-daemon-or-script.py" ]