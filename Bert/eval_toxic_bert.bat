set BERT_BASE_DIR=G:\python\DeepLearning\Learning_Cloud\Bert\uncased_L-12_H-768_A-12

set MY_DATASET=G:\python\DeepLearning\Learning_Cloud\temp\cn_nlp\corpus\kaggle_Jigsaw

set OUT_DIR=/tmp/toxic_model/

python run_classifier.py ^
  --data_dir=%MY_DATASET% ^
  --task_name=toxic ^
  --vocab_file=%BERT_BASE_DIR%/vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json ^
  --output_dir=%OUT_DIR% ^
  --do_eval=true ^
  --init_checkpoint=%OUT_DIR% ^
  --max_seq_length=128 ^
