set BERT_BASE_DIR=G:\python\DeepLearning\Learning_Cloud\Bert\uncased_L-12_H-768_A-12

set MY_DATASET=G:\python\DeepLearning\Learning_Cloud\temp\cn_nlp\corpus\kaggle_Jigsaw

set OUT_DIR=/tmp/toxic_model/

python run_classifier.py ^
  --data_dir=%MY_DATASET% ^
  --task_name=toxic ^
  --vocab_file=%BERT_BASE_DIR%/vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json ^
  --output_dir=%OUT_DIR% ^
  --do_train=true ^
  --do_eval=true ^
  --init_checkpoint=%BERT_BASE_DIR%/bert_model.ckpt ^
  --max_seq_length=128 ^
  --train_batch_size=32 ^
  --learning_rate=5e-5^
  --num_train_epochs=1.0 
