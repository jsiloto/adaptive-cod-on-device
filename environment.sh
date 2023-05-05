#!/bin/bash


sudo apt-get install -y bison byacc flex autotools-dev automake
sudo apt-get install -y libtool libdbus-1-dev libglib2.0-dev
sudo apt-get install -y virtualenv
sudo apt-get install -y python3-dev

mkdir workspace
cd workspace


######################
git clone https://github.com/bluez/bluez.git
cd bluez
git checkout 4.101
./bootstrap
./configure --prefix=/usr --mandir=/usr/share/man \
		--sysconfdir=/etc --localstatedir=/var --libexecdir=/lib
make -j4
sudo make install
sudo systemctl daemon-reload
sudo service bluetooth restart


cd ..
git clone https://github.com/jsiloto/adaptive-cod-on-device.git
cd adaptive-cod-on-device
git checkout journal
virtualenv -p python3 env
source env/bin/activate
pip install -U pip
pip install -r requirements.txt
mkdir models



sdptool add SP
sudo hciconfig hci0 piscan