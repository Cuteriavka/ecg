#! /bin/bash
#
# file: setup.sh
#
# This bash script performs any setup necessary in order to test your
# entry.  It is run only once, before running any other code belonging
# to your entry.

set -e
set -o pipefail


# challenge system has not enough space in /tmp, apparently
export TMPDIR=$HOME/temp
mkdir -p $TMPDIR

virtualenv -p python3.6 myenv
source D:/Works/Python_workspace/tens_ecg/ecg/examples/cinc17/entry/myenv/Scripts/activate || true
#pip install --upgrade packages/setuptools-39.2.0.zip
#pip install --upgrade packages/wheel-0.31.1.tar.gz
#pip install --upgrade packages/pip-10.0.1.tar.gz
#(cd packages;
# pip install \
#         absl-py-0.2.2.tar.gz \
#         astor-0.6.2-py2.py3-none-any.whl \
#         backports.weakref-1.0.post1-py2.py3-none-any.whl \
#         bleach-1.5.0-py2.py3-none-any.whl \
#         enum34-1.1.6-py2-none-any.whl \
#         funcsigs-1.0.2-py2.py3-none-any.whl \
#         futures-3.2.0-py2-none-any.whl \
#         gast-0.2.0.tar.gz \
#         grpcio-1.12.1-cp27-cp27mu-manylinux1_x86_64.whl \
#         h5py-2.8.0rc1-cp27-cp27mu-manylinux1_x86_64.whl \
#         html5lib-0.9999999.tar.gz \
#         Keras-2.1.6-py2.py3-none-any.whl \
#         linecache2-1.0.0-py2.py3-none-any.whl \
#         Markdown-2.6.11-py2.py3-none-any.whl \
#         mock-2.0.0-py2.py3-none-any.whl \
#         numpy-1.14.3-cp27-cp27mu-manylinux1_x86_64.whl \
#         pbr-4.0.4-py2.py3-none-any.whl \
#         protobuf-3.5.2.post1-cp27-cp27mu-manylinux1_x86_64.whl \
#         PyYAML-3.12.tar.gz \
#         scipy-1.1.0-cp27-cp27mu-manylinux1_x86_64.whl \
#         six-1.11.0-py2.py3-none-any.whl \
#         tensorboard-1.8.0-py2-none-any.whl \
#         tensorflow-1.8.0-cp27-cp27mu-manylinux1_x86_64.whl \
#         termcolor-1.1.0.tar.gz \
#         tqdm-4.23.4-py2.py3-none-any.whl \
#         traceback2-1.4.0-py2.py3-none-any.whl \
#         unittest2-1.1.0-py2.py3-none-any.whl \
#         Werkzeug-0.14.1-py2.py3-none-any.whl)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple absl-py==0.2.2
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple astor==0.6.2
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple backports.weakref==1.0.post1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple bleach==1.5.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple enum34==1.1.8
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple funcsigs==1.0.2
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple futures==3.1.1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gast==0.2.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple grpcio==1.12.1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple h5py==2.8.0rc1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple html5lib==0.9999999
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Keras==2.1.6
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple linecache2==1.0.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Markdown==2.6.11
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mock==2.0.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.14.3
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pbr==4.0.4
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple protobuf==3.6.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyYAML==3.12
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple setuptools==50.3.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy==1.1.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple six==1.11.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard==1.8.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.8.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.8.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple termcolor==1.1.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm==4.23.4
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple traceback2==1.4.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple unittest2==1.1.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Werkzeug==0.14.1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wheel==0.31.1

