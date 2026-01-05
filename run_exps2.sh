#!/bin/bash

for train_ds in Quartets Beethoven Haydn Mozart; do
    python -u train.py --ds_name $train_ds --model_type transformer --batch_size 1 --patience 5 --attn_window 100 --input_feature ihcogram --encoder_dropout_p 0.0 --decoder_dropout_p 0.0 --position_encoding_dropout_p 0.0

    for test_ds in Quartets Beethoven Haydn Mozart; do
        if [ $train_ds != $test_ds ]; then
            python -u test.py --ds_name $test_ds --model_type transformer --checkpoint_path weights/transformer/$train_ds.ckpt --input_feature ihcogram
        fi
    done
done
