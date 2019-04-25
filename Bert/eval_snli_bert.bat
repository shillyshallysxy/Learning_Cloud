set BERT_BASE_DIR=G:\python\DeepLearning\Learning_Cloud\Bert\uncased_L-12_H-768_A-12

set MY_DATASET=G:\python\DeepLearning\Learning_Cloud\temp\cn_nlp\corpus\SNLI\snli_1.0

set OUT_DIR=/tmp/snli_model/

python run_classifier.py ^
  --data_dir=%MY_DATASET% ^
  --task_name=snli ^
  --vocab_file=%BERT_BASE_DIR%/vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json ^
  --output_dir=%OUT_DIR% ^
  --do_eval=True ^
  --init_checkpoint=%OUT_DIR% ^
  --max_seq_length=128
