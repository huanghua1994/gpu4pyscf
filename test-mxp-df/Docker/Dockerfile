ARG image
FROM ${image}

ARG uid=1924
ARG gid=1112
ARG username=appuser
ARG groupname=appgroup

RUN apt update && apt upgrade -y
RUN apt install -y sudo htop nvtop tree dos2unix build-essential gfortran gdb wget curl git cmake python3 pip
RUN apt clean && apt autoremove -y

RUN groupadd --gid ${gid} ${groupname}
RUN useradd --uid ${uid} --gid ${gid} --shell /bin/bash --create-home ${username}
RUN usermod -aG sudo ${username}
RUN echo "${username} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${username}
RUN chmod 0440 /etc/sudoers.d/${username}

USER ${username}
COPY --chown=${username} ./bashrc.txt /home/${username}/.bashrc
COPY --chown=${username} ./vimrc.txt  /home/${username}/.vimrc
RUN sudo pip install --break-system-packages numpy scipy
RUN sudo pip install --break-system-packages --prefer-binary pyscf
RUN sudo pip install --break-system-packages cupy-cuda12x cutensor-cu12


RUN cd /home/${username} && git clone https://github.com/huanghua1994/gpu4pyscf.git
WORKDIR /home/${username}/gpu4pyscf
RUN cd gpu4pyscf && git checkout mxp-df
RUN cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf
RUN cmake --build build/temp.gpu4pyscf -j 16
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/home/'${username}'/gpu4pyscf"' >> /home/${username}/.bashrc

WORKDIR /home/${username}/gpu4pyscf/test-mxp-df
