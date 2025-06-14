from typing import List, Optional

import torch

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
#from gluonts.torch.util import copy_parameters
from missingutils import copy_parameters
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    RemoveFields,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)

#from pts import Trainer
from trainer import Trainer
#from pts.model import PyTorchEstimator
from estimator import PyTorchEstimator
from pts.model.utils import get_module_forward_input_names
from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)

from localpts.model.transformer_tempflow.transformer_tempflow_network import (
    TransformerTempFlowTrainingNetwork,
    TransformerTempFlowPredictionNetwork,
)


class TransformerTempFlowEstimator(PyTorchEstimator):
    @validated()
    def __init__(
        self,
        input_size: int,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        d_model: int = 32,
        dim_feedforward_scale: int = 4,
        act_type: str = "gelu",
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        flow_type="RealNVP",
        n_blocks=3,
        hidden_size=100,
        n_hidden=2,
        conditioning_length: int = 200,
        dequantize: bool = False,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        **kwargs,
    ) -> None:
        super().__init__(trainer=trainer, **kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

        self.input_size = input_size
        self.prediction_length = prediction_length
        self.target_dim = target_dim

        self.d_model = d_model
        self.num_heads = num_heads
        self.act_type = act_type
        self.dim_feedforward_scale = dim_feedforward_scale
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        
        self.use_feat_dynamic_real = use_feat_dynamic_real

        self.flow_type = flow_type
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.conditioning_length = conditioning_length
        self.dequantize = dequantize

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=freq)
        )

        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        
        return Chain(
            [
                RemoveFields(field_names=remove_field_names),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def create_training_network(
        self, device: torch.device
    ) -> TransformerTempFlowTrainingNetwork:
        return TransformerTempFlowTrainingNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_heads=self.num_heads,
            act_type=self.act_type,
            d_model=self.d_model,
            dim_feedforward_scale=self.dim_feedforward_scale,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            flow_type=self.flow_type,
            n_blocks=self.n_blocks,
            hidden_size=self.hidden_size,
            n_hidden=self.n_hidden,
            conditioning_length=self.conditioning_length,
            dequantize=self.dequantize,
        ).to(device)

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: TransformerTempFlowTrainingNetwork,
        device: torch.device,
    ) -> Predictor:
        prediction_network = TransformerTempFlowPredictionNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_heads=self.num_heads,
            act_type=self.act_type,
            d_model=self.d_model,
            dim_feedforward_scale=self.dim_feedforward_scale,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            flow_type=self.flow_type,
            n_blocks=self.n_blocks,
            hidden_size=self.hidden_size,
            n_hidden=self.n_hidden,
            conditioning_length=self.conditioning_length,
            dequantize=self.dequantize,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device)

        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
