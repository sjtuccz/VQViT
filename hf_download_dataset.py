import openxlab
openxlab.login(ak="lemxwd0yomwazl15pavw", sk="egpnmqbnkrzzdg3vrwgkpz5qayxep05241owv6ay") 
from openxlab.dataset import info
info(dataset_repo='OpenDataLab/ImageNet-Sketch') 
from openxlab.dataset import get 
get(dataset_repo='OpenDataLab/ImageNet-R', target_path='/home/mulan/ccz/imagenet_r/')
# get(dataset_repo='OpenDataLab/ImageNet-Sketch', target_path='/home/mulan/ccz/imagenet_sketch/')
# from openxlab.dataset import download
# download(dataset_repo='OpenDataLab/ImageNet-Sketch',source_path='/README.md', target_path='/home/mulan/ccz/imagenet_sketch/') 