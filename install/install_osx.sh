
# Install homebrew
ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"

sudo easy_install pip
brew tap homebrew/science
brew tap homebrew/python

# Directions taken from 
# http://joernhees.de/blog/2013/06/08/mac-os-x-10-8-scientific-python-with-homebrew/

# set up some taps and update brew
brew tap homebrew/science # a lot of cool formulae for scientific tools
brew tap homebrew/python # numpy, scipy
brew update && brew upgrade

# install a brewed python
brew install python

# install openblas (otherwise scipy's arpack tests will fail)
brew install openblas

# install nose (unittests & doctests on steroids)
pip install virtualenv nose

# install numpy and scipy
brew install numpy --with-openblas # bug in Accelerate framework < Mac OS X 10.9
brew install scipy --with-openblas # bug in Accelerate framework < Mac OS X 10.9

# test the scipy install
brew test scipy

sudo pip install ipython
sudo pip install pandas nltk matplotlib sympy q

# ipython and notebook support
brew install zmq
pip install ipython[zmq,qtconsole,notebook,test]

ARGS='ARCHFLAGS="-Wno-error=unused-command-line-argument-hard-error-in-future"'

sudo ARCHFLAGS="-Wno-error=unused-command-line-argument-hard-error-in-future" pip install numexpr
sudo ARCHFLAGS="-Wno-error=unused-command-line-argument-hard-error-in-future" pip install -U celery

brew install hdf5
brew install rabbitmq
