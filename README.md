# flow_physics

Unifying Predictions of Deterministic and Stochastic Physics in Mesh-reduced Space with Sequential Flow Generative Model

## 0. Environment Setup

enviroment setup: 
```
conda env create -f environment.yml
```
```
conda activate test_flow
```
## Cases
flow past cylinder


## 1. Training
To train the demo, run the following command:
```
python main.py --token --mu_size 2 --train 1 --epoch 20000 --dataset 'cylinder' --conditioning_length 1024 --input_size 1024 --hidden_size 1024 --n_hidden 2 --batchsize 2 --num_encoder_layers 2 --num_decoder_layers 1 --lr 1e-4

```

To draw the latent for les2d, run the following command
```

python main.py --mu_size 5 --dataset les2d --train 1 --conditioning_length 1024 --hidden_size 1024 --input_size 1024 --num_heads 8 --d_model 1024 --dim_feedforward_scale 1 --num_encoder_layers 1 --num_decoder_layers 1 --batchsize 5 --shuffle False --transfer_flag 1 --transfer_epoch 220000 --modelref transfer220000_transformer_les2d_normalized_cond1024_flow1024_5cases_1024pt_2nhidden_heads8_dmodel024_dim_scale1_nencode1_ndecode1_batch

```

To test the demo run the following:

``` 
python main.py --token --mu_size 2 --train 0 --test_epoch 20000 --dataset 'cylinder' --conditioning_length 1024 --input_size 1024 --hidden_size 1024 --n_hidden 2 --batchsize 2 --num_encoder_layers 2 --num_decoder_layers 1 --lr 1e-4 
```

```
To train the les 2d 30 cases 3 decoders run the following:
python main.py --mu_size 30 --dataset les2d30 --train 1 --conditioning_length 1024 --hidden_size 1024 --input_size 1024 --num_heads 8 --d_model 1024 --dim_feedforward_scale 1 --num_encoder_layers 2 --num_decoder_layers 3 --batchsize 5 --shuffle True --modelref transformer_les2d30_normalized_cond1024_flow1024_30cases_1024pt_2nhidden_heads8_dmodel024_dim_scale1_nencode2_ndecode3_batch5 --lr 1e-4 --device cuda:0 --epoch 240000
```


```
To tes the les 2d 30 cases 3 decoders run the following:

python main.py --mu_size 30 --dataset les2d30 --train 0 --conditioning_length 1024 --hidden_size 1024 --input_size 1024 --num_heads 8 --d_model 1024 --dim_feedforward_scale 1 --num_encoder_layers 2 --num_decoder_layers 3 --batchsize 5 --shuffle True --modelref transformer_les2d30_normalized_cond1024_flow1024_30cases_1024pt_2nhidden_heads8_dmodel024_dim_scale1_nencode2_ndecode3_batch5 --lr 1e-4 --test_epoch 240000 --test_samples 5 --history_length 2 --prediction_length 238
```

