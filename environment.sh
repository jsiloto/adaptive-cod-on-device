#!/bin/bash

sudo apt-get update
sudo apt-get install -y bison byacc flex autotools-dev automake
sudo apt-get install -y vim git

cd ..
git clone https://github.com/bluez/bluez.git

#sdptool add SP
#
#sudo hciconfig hci0 piscan