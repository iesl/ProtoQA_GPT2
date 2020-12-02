conda create -n myenv python=3.6
source activate protoqa
git clone git@github.com:iesl/protoqa-evaluator.git
pip install -e protoqa-evaluator
conda install pytorch=1.4.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
git clone git@github.com:huggingface/transformers.git
cd transformers
git checkout tags/v2.1.1
pip install -e .
pip install tensorboardX
cd ../
mkdir pre_trained_model
