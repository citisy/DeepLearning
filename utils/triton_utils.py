import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
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
    def __init__(self, url, verbose=False, logger=None):
        self.client = InferenceServerClient(url=url, verbose=False)
        self.verbose = verbose
        self.logger = get_logger(logger)
        self.model_configs = {}
        self.init()

    def init(self):
        model_info = self.client.get_model_repository_index()

        for info in model_info:
            name = info['name']
            version = info['version']
            state = info['state']

            if state != 'READY':
                self.logger.warning(f'{name}:{version}({state}) is not ready, pls check!')
                continue

            self.model_configs[name, version] = self.client.get_model_config(name, version)

        if self.verbose:
            self.logger.info(self.model_configs)

    def async_infer(self, *inputs: 'np.ndarray', model_name, model_version):
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

    def async_get(self, async_req):
        result = async_req.get_result()
        response = result.get_response()

        outputs = {}
        for output in response['outputs']:
            name = output['name']
            outputs[name] = result.as_numpy(name)

        return outputs
