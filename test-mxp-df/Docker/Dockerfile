ARG image
FROM ${image}

ARG uid
ARG gid
ARG username
ARG groupname

RUN apt update
RUN apt install -y sudo vim htop nvtop tree dos2unix build-essential gfortran gdb wget curl git cmake python3 pip
RUN apt clean && apt autoremove -y

RUN pip install --break-system-packages numpy scipy
RUN pip install --break-system-packages --prefer-binary pyscf
RUN pip install --break-system-packages cupy-cuda12x cutensor-cu12

RUN addgroup --gid ${uid} ${username}
RUN groupmod --gid 30 dip
RUN groupmod --gid 3000 dip
RUN addgroup --gid ${gid} ${groupname}
RUN adduser --disabled-password --gecos GECOS -u ${uid} -gid ${gid} ${username}
RUN adduser ${username} sudo
RUN usermod -a -G ${groupname} ${username}
RUN usermod -a -G ${groupname} root
RUN usermod -a -G root ${username}
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${username}
COPY --chown=${username} ./bashrc.txt /home/${username}/.bashrc
COPY --chown=${username} ./vimrc.txt  /home/${username}/.vimrc

#RUN echo 'export PYTHONPATH="${PYTHONPATH}:/gpu4pyscf"' >> /home/${username}/.bashrc
#WORKDIR /gpu4pyscf
