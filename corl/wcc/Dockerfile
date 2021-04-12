FROM tensorflow/tensorflow:2.4.1-gpu

# set timezone to Asia/Shanghai
#ENV TZ Asia/Shanghai

#RUN echo $TZ > /etc/timezone && \
#     apt-get update && apt-get install -y tzdata && \
#     rm /etc/localtime && \
#     ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
#     dpkg-reconfigure -f noninteractive tzdata && \
#     apt-get clean

# install timedatectl & dependencies, and enable time synchronization
#RUN apt-get install -y systemd dbus
#RUN timedatectl set-ntp on

# upgrade pip and install requirements
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt;
RUN rm -rf requirements.txt