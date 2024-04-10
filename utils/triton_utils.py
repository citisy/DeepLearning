import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput, InferAsyncRequest
from .log_utils import get_logger

# https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes
datatypes = {
    'Config': ['TYPE_BOOL', 'TYPE_UINT8', 'TYPE_UINT16', 'TYPE_UINT32', 'TYPE_UINT64', 'TYPE_INT8', 'TYPE_INT16', 'TYPE_INT32', 'TYPE_INT64', 'TYPE_FP16', 'TYPE_FP32', 'TYPE_FP64', 'TYPE_STRING', 'TYPE_BF16'],
    'API': ['BOOL', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'INT8', 'INT16', 'INT32', 'INT64', 'FP16', 'FP32', 'FP64', 'BYTES', 'BF16'],
    'TensorRT': ['kBOOL', 'kUINT8', '', '', '', 'kINT8', '', 'kINT32', '', 'kHALF', 'kFLOAT', '', '', ''],
    'TensorFlow': ['DT_BOOL', 'DT_UINT8', 'DT_UINT16', 'DT_UINT32', 'DT_UINT64', 'DT_INT8', 'DT_INT16', 'DT_INT32', 'DT_INT64', 'DT_HALF', 'DT_FLOAT', 'DT_DOUBLE', 'DT_STRING', ''],
    'ONNX': ['BOOL', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'INT8', 'INT16', 'INT32', 'INT64', 'FLOAT16', 'FLOAT', 'DOUBLE', 'STRING', ''],
    'PyTorch': ['kBool', 'kByte', '', '', '', 'kChar', 'kShort', 'kInt', 'kLong', '', 'kFloat', 'kDouble', '', ''],
    'NumPy': ['bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'dtype(object)', '']
}


class Requests:
    def __init__(self, url, verbose=False, logger=None, **kwargs):
        self.client = InferenceServerClient(url=url, verbose=False, **kwargs)
        self.verbose = verbose
        self.logger = get_logger(logger)
        self.model_configs = {}
        self.model_versions = {}
        # self.init()

    def init(self):
        model_info = self.client.get_model_repository_index()

        for info in model_info:
            name = info['name']
            version = info['version']
            state = info['state']

            if state != 'READY':
                self.logger.warning(f'{name}:{version}({state}) is not ready, pls check!')
                continue

            self.model_versions[name] = version
            self.model_configs[name, version] = self.client.get_model_config(name, version)

        if self.verbose:
            self.logger.info(self.model_configs)

    def async_infer(self, *inputs: 'np.ndarray', model_name, model_version=None):
        if not self.model_versions:
            # note, in some version of triton, it will be got some unknown exceptions when init early
            # so init when using
            self.init()

        model_version = model_version or self.model_versions.get(model_name)
        assert (model_name, model_version) in self.model_configs, \
            f'Got {model_name = } and {model_version = }, where the keys is {self.model_configs.keys()}, pls check'

        model_config = self.model_configs[model_name, model_version]

        _inputs = []
        for cfg, i in zip(model_config['input'], inputs):
            dtype = cfg['data_type']
            dtype = datatypes['API'][datatypes['Config'].index(dtype)]

            _input = InferInput(cfg['name'], i.shape, dtype)
            _input.set_data_from_numpy(i)
            _inputs.append(_input)

        _outputs = []
        for cfg in model_config['output']:
            _output = InferRequestedOutput(cfg['name'])
            _outputs.append(_output)

        async_req = self.client.async_infer(
            model_name=model_name,
            model_version=model_version,
            inputs=_inputs,
            outputs=_outputs
        )

        return async_req

    @staticmethod
    def async_get(async_req: InferAsyncRequest):
        result = async_req.get_result()
        response = result.get_response()

        outputs = {}
        for output in response['outputs']:
            name = output['name']
            outputs[name] = result.as_numpy(name)

        return outputs


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    this is an example
    refer to: https://github.com/triton-inference-server/python_backend
    """

    def initialize(self, args):
        import triton_python_backend_utils as pb_utils
        import json

        # get configs
        # configs can be found in `config.pbtxt`
        self.model_config = model_config = json.loads(args['model_config'])

        self.input0_config = pb_utils.get_output_config_by_name(model_config, "INPUT0")
        self.input0_dtype = pb_utils.triton_string_to_numpy(self.input0_config['data_type'])
        self.output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(self.output0_config['data_type'])
        ...

        # init model
        self.model = ...

    def execute(self, requests):
        import triton_python_backend_utils as pb_utils

        responses = []
        for request in requests:
            # get inputs
            in_0 = pb_utils.get_input_tensor_by_name(request, 'INPUT0')
            ...

            # get outputs from model inference
            out_0, *outs = self.model(in_0, ...)

            out_tensor_0 = pb_utils.Tensor('OUTPUT0', out_0.astype(self.output0_dtype))
            ...

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, ...])

            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
