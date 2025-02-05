## 가중치 이동
mkdir trained_checkpoint  
cp ...  

## 환경설정
conda new -n cloth_seg python=3.10.11  
conda activate cloth_seg  
pip install -r requirements.txt  

## 입력 이미지, 출력 이미지 디렉토리 생성 및 이미지 이동
mkdir input_images  
mkdir output_images  

## 모델 체크
python infer_b2.py  
python infer_u2.py  

## payload 만들기
./payloads/your_payload.json  

## 생성할 prompt 준비하기, .txt파일당 개별임
./prompts/0000.txt
./prompts/0001.txt

## webui런
bash run_webui_0.sh  

## 파이프 실행
python python pipe_human-data_0.py  