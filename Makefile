dep:
	pip install -r requirements.txt

model_checkpoint:
	gdown --output checkpoint_u2net.pth 1aS4gB2UucK1twuU5ROg_MYRIxm3jmk1v

infer_cpu:
	PYTHONPATH=. DEVICE=cpu python infer.py

infer_gpu:
	PYTHONPATH=. python cloth_segmentation/infer.py
