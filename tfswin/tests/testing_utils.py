import numpy as np
import tensorflow as tf
from keras import backend, models, layers
from keras.src.testing_infra.test_utils import _thread_local_data, should_run_eagerly
from tensorflow.python.framework import tensor_shape, test_util
from tensorflow.python.util import tf_inspect


@test_util.disable_cudnn_autotune
def layer_multi_io_test(
        layer_cls, kwargs=None, input_shapes=None, input_dtypes=None,
        input_datas=None, expected_outputs=None, expected_output_dtypes=None,
        expected_output_shapes=None, validate_training=True, adapt_data=None):
    """Test routine for a layer with multiple inputs and outputs.

    Arguments:
      layer_cls: Layer class object.
      kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
      input_shapes: Input shape tuples.
      input_dtypes: Data types of the input data.
      input_datas: Numpy arrays of input data.
      expected_outputs: Numpy arrays of the expected output.
      expected_output_dtypes: Data types expected for the output.
      expected_output_shapes: Shape tuples for the expected output shapes.
      validate_training: Whether to attempt to validate training on this layer.
        This might be set to False for non-differentiable layers that output
        string or integer values.
      adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
        be tested for this layer. This is only relevant for PreprocessingLayers.

    Returns:
      The output data (Numpy array) returned by the layer, for additional
      checks to be done by the calling code.

    Raises:
      ValueError: if `input_shape is None`.
    """
    if input_shapes is not None:
        if not isinstance(input_shapes, (list, tuple)):
            raise ValueError(
                'A list of shape tuples expected for input_shapes')
        for input_shape in input_shapes:
            if not isinstance(input_shape, (list, tuple)):
                raise ValueError(
                    'A list of shape tuples expected for input_shapes')
            for shape_dim in input_shape:
                if not isinstance(shape_dim, (int, type(None))):
                    raise ValueError(
                        'Only integer and None values allowed in input_shapes')

    if input_dtypes is not None:
        if not isinstance(input_dtypes, (list, tuple)):
            raise ValueError(
                'A list data type names expected for input_dtypes')
        for input_dtype in input_dtypes:
            if not isinstance(input_dtype, str):
                raise ValueError(
                    'Only string values allowed in input_dtypes')

    if input_datas is not None:
        if not isinstance(input_datas, (list, tuple)):
            raise ValueError('A list of numpy arrays expected for input_datas')
        for input_data in input_datas:
            if not isinstance(input_data, np.ndarray):
                raise ValueError(
                    'A list of numpy arrays expected for input_datas')

    output_size = -1
    if expected_outputs is not None:
        if not isinstance(expected_outputs, (list, tuple)):
            raise ValueError(
                'A list of numpy arrays expected for expected_outputs')
        for expected_output in expected_outputs:
            if not isinstance(expected_output, np.ndarray):
                raise ValueError(
                    'A list of numpy arrays expected for expected_outputs')
        output_size = max(output_size, len(expected_outputs))
    if expected_output_dtypes is not None:
        if not isinstance(expected_output_dtypes, (list, tuple)):
            raise ValueError(
                'A list data type names expected for expected_output_dtypes')
        for expected_output_dtype in expected_output_dtypes:
            if not isinstance(expected_output_dtype, str):
                raise ValueError(
                    'Only string values allowed in expected_output_dtypes')
        output_size = max(output_size, len(expected_output_dtypes))
    if expected_output_shapes is not None:
        if not isinstance(expected_output_shapes, (list, tuple)):
            raise ValueError(
                'A list of shape tuples expected for expected_output_shapes')
        for expected_output_shape in expected_output_shapes:
            if not isinstance(expected_output_shape, (list, tuple)):
                raise ValueError(
                    'A list of shape tuples expected for expected_output_shapes')
            for shape_dim in expected_output_shape:
                if not isinstance(shape_dim, (int, type(None))):
                    raise ValueError(
                        'Only integer and None values allowed in '
                        'expected_output_shapes')
        output_size = max(output_size, len(expected_output_shapes))

    if expected_outputs is not None and \
            expected_output_dtypes is not None and \
            len(expected_outputs) != len(expected_output_dtypes):
        raise ValueError(
            'Sizes of "expected_outputs" and "expected_output_dtypes" '
            'should be equal if both provided')
    if expected_outputs is not None and \
            expected_output_shapes is not None and \
            len(expected_outputs) != len(expected_output_shapes):
        raise ValueError(
            'Sizes of "expected_outputs" and "expected_output_shapes" '
            'should be equal if both provided')
    if expected_output_dtypes is not None and \
            expected_output_shapes is not None and \
            len(expected_output_dtypes) != len(expected_output_shapes):
        raise ValueError(
            'Sizes of "expected_output_dtypes" and "expected_output_shapes" '
            'should be equal if both provided')

    if 0 >= output_size:
        raise ValueError(
            'Could not determine number of outputs. Provide at least one of: '
            '"expected_output_dtypes" or "expected_output_shapes" or '
            '"expected_outputs"')

    input_size = -1
    if input_datas is None:
        if input_shapes is None:
            raise ValueError(
                'Either input_shapes or input_datas should be provided')
        input_size = len(input_shapes)
        if not input_dtypes:
            input_dtypes = ['float32'] * input_size

        input_datas = []
        input_data_shapes = [list(input_shape) for input_shape in input_shapes]
        for i, input_data_shape in enumerate(input_data_shapes):
            for j, e in enumerate(input_data_shape):
                if e is None:
                    input_data_shape[j] = np.random.randint(1, 4)
            input_data = 10 * np.random.random(input_data_shape)
            if input_dtypes[i][:5] == 'float':
                input_data -= 0.5
            input_data = input_data.astype(input_dtypes[i])
            input_datas.append(input_data)
    elif input_shapes is None:
        input_size = len(input_datas)
        input_shapes = [input_data.shape for input_data in input_datas]
    else:
        if len(input_datas) != len(input_shapes):
            raise ValueError(
                'Sizes of "input_datas" and "input_shapes" should be equal '
                'if both provided')
        for input_data, input_shape in zip(input_datas, input_shapes):
            if len(input_data.shape) != len(input_shape) or \
                    not np.all(np.equal(input_data.shape, input_shape)):
                raise ValueError(
                    'Shapes of "input_datas" and values in "input_shapes" '
                    'should be equal if both provided')
        input_size = len(input_datas)

    if 0 >= input_size:
        raise ValueError('Wrong number of inputs')

    if input_dtypes is None:
        input_dtypes = [input_data.dtype for input_data in input_datas]
    if expected_output_dtypes is None:
        expected_output_dtypes = input_dtypes[:1] * output_size

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
        kwargs['weights'] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    xs = [
        layers.Input(shape=input_shapes[i][1:], dtype=input_dtypes[i])
        for i in range(input_size)]
    _y = layer(_squize(xs, input_size))
    ys = _expand(_y, output_size)

    if 1 == output_size and isinstance(_y, (list, tuple)):
        raise AssertionError(
            'When testing layer {}, for inputs {}, found {} outputs but '
            'expected to find 1.\nFull kwargs: {}'.format(
                layer_cls.__name__, xs, len(_y), kwargs))
    elif 1 < output_size and not isinstance(_y, (list, tuple)):
        raise AssertionError(
            'When testing layer {}, for inputs {}, found {} outputs but '
            'expected to find {} outputs.\nFull kwargs: {}'.format(
                layer_cls.__name__, xs, _y, output_size, kwargs))

    try:
        _assert_dtypes(ys, expected_output_dtypes)
    except AssertionError:
        raise AssertionError(
            'When testing layer {}, for inputs {}, found output dtypes={} '
            'but expected to find {}.\nFull kwargs: {}'.format(
                layer_cls.__name__, xs, [backend.dtype(yi) for yi in ys],
                expected_output_dtypes, kwargs))

    if expected_output_shapes is not None:
        expected_shapes = [
            tensor_shape.TensorShape(sh)
            for sh in expected_output_shapes]
        actual_shapes = [yi.shape for yi in ys]

        try:
            _assert_shapes(expected_shapes, actual_shapes)
        except AssertionError:
            raise AssertionError(
                'When testing layer {}, for inputs {}, found output_shapes={} '
                'but expected to find {}.\nFull kwargs: {}'.format(
                    layer_cls.__name__, xs, actual_shapes,
                    expected_shapes, kwargs))

    # check shape inference
    model = models.Model(_squize(xs, input_size), _squize(ys, output_size))

    compute_input_shapes = _squize([
        tensor_shape.TensorShape(sh) for sh in input_shapes], input_size)
    computed_output_shapes = _expand(layer.compute_output_shape(
        compute_input_shapes), output_size)
    computed_output_shapes = [
        tuple(sh.as_list()) for sh in computed_output_shapes]

    compute_input_signatures = _squize([
        tf.TensorSpec(shape=input_shapes[i], dtype=input_dtypes[i])
        for i in range(input_size)], input_size)
    computed_output_signatures = _expand(layer.compute_output_signature(
        compute_input_signatures), output_size)
    computed_output_signature_shapes = [
        cs.shape for cs in computed_output_signatures]
    computed_output_signature_dtypes = [
        cs.dtype for cs in computed_output_signatures]

    actual_outputs = _expand(model.predict(_squize(
        input_datas, input_size)), output_size)
    actual_output_shapes = [ao.shape for ao in actual_outputs]
    actual_output_dtypes = [ao.dtype for ao in actual_outputs]

    try:
        _assert_shapes(computed_output_shapes, actual_output_shapes)
    except AssertionError:
        raise AssertionError(
            'When testing layer {}, for inputs {}, found output_shapes={} '
            'but expected to find {}.\nFull kwargs: {}'.format(
                layer_cls.__name__, xs, actual_output_shapes,
                computed_output_shapes, kwargs))

    try:
        _assert_shapes(computed_output_signature_shapes, actual_output_shapes)
    except AssertionError:
        raise AssertionError(
            'When testing layer {}, for inputs {}, found output_shapes={} '
            'but expected to find {}.\nFull kwargs: {}'.format(
                layer_cls.__name__, xs, actual_output_shapes,
                computed_output_signatures, kwargs))

    try:
        _assert_dtypes(computed_output_signatures, actual_output_dtypes)
    except AssertionError:
        raise AssertionError(
            'When testing layer {}, for inputs {}, found output dtypes={} '
            'but expected to find {}.\nFull kwargs: {}'.format(
                layer_cls.__name__, xs, actual_output_dtypes,
                computed_output_signature_dtypes, kwargs))

    if expected_outputs is not None:
        for i in range(output_size):
            np.testing.assert_allclose(
                actual_outputs[i], expected_outputs[i], rtol=1e-3, atol=1e-6)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = models.Model.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        outputs = _expand(recovered_model.predict(_squize(
            input_datas, input_size)), output_size)
        for i in range(output_size):
            np.testing.assert_allclose(
                outputs[i], actual_outputs[i], rtol=1e-3, atol=1e-6)

    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # See b/120160788 for more details. This should be mitigated after 2.0.
    if validate_training:
        _x = _squize(xs, input_size)
        model = models.Model(_x, layer(_x))
        if _thread_local_data.run_eagerly is not None:
            model.compile('rmsprop', 'mse', weighted_metrics=['acc'], run_eagerly=should_run_eagerly())
        else:
            model.compile('rmsprop', 'mse', weighted_metrics=['acc'])
        model.train_on_batch(
            _squize(input_datas, input_size),
            _squize(actual_outputs, output_size))

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    # Sequential model does not supports multiple inputs or outputs
    #
    # # test as first layer in Sequential API
    # layer_config = layer.get_config()
    # layer_config['batch_input_shape'] = input_shape
    # layer = layer.__class__.from_config(layer_config)
    #
    # Sequential model does not supports multiple inputs or outputs
    # model = models.Sequential()
    # model.add(layers.Input(shape=input_shape[1:], dtype=input_dtype))
    # model.add(layer)
    # actual_output = model.predict(input_data)
    # actual_output_shape = actual_output.shape
    # for expected_dim, actual_dim in zip(computed_output_shape,
    #                                     actual_output_shape):
    #     if expected_dim is not None:
    #         if expected_dim != actual_dim:
    #             raise AssertionError(
    #                 'When testing layer {} **after deserialization**, for '
    #                 'input {}, found output_shape={} but expected to find '
    #                 'inferred shape {}.\nFull kwargs: {}'.format(
    #                     layer_cls.__name__, x, actual_output_shape,
    #                     computed_output_shape, kwargs))
    #
    #
    # if expected_output is not None:
    #     np.testing.assert_allclose(actual_output, expected_output,
    #                                rtol=1e-3, atol=1e-6)
    #
    # # test serialization, weight setting at model level
    # model_config = model.get_config()
    # recovered_model = models.Sequential.from_config(model_config)
    # if model.weights:
    #     weights = model.get_weights()
    #     recovered_model.set_weights(weights)
    #     output = recovered_model.predict(input_data)
    #     np.testing.assert_allclose(
    #         output, actual_output, rtol=1e-3, atol=1e-6)

    # for further checks in the caller function
    return _squize(actual_outputs, output_size)


def _squize(data, size):
    if size > 1 and (not isinstance(data, (list, tuple)) or not len(data)):
        raise ValueError('Wrong "data" value')

    return data[0] if 1 == size else data


def _expand(data, size):
    return [data] if 1 == size else data


def _assert_dtypes(tensors, expected_dtypes):
    if not isinstance(tensors, (list, tuple)):
        raise ValueError('A list of tensors should be provided for "tensors"')
    if not isinstance(expected_dtypes, (list, tuple)):
        raise ValueError(
            'A list of dtype names should be provided for "expected_dtypes"')
    if len(tensors) != len(expected_dtypes):
        raise ValueError(
            'Sizes of "tensors" and corresponding "expected_dtypes" '
            'should be equal')

    for tensor, expected_dtype in zip(tensors, expected_dtypes):
        if backend.dtype(tensor) != expected_dtype:
            raise AssertionError('Wrong dtype')


def _assert_shapes(expected_shapes, actual_shapes):
    if not isinstance(expected_shapes, list):
        raise ValueError(
            'A list of shapes should be provided for "expected_shapes"')
    if not isinstance(actual_shapes, list):
        raise ValueError(
            'A list of shapes should be provided for "actual_shapes"')
    if len(expected_shapes) != len(actual_shapes):
        raise ValueError(
            'Sizes of "expected_shapes" and corresponding "actual_shapes" '
            'should be equal')

    for expected_shape, actual_shape in zip(expected_shapes, actual_shapes):
        _assert_shape(expected_shape, actual_shape)


def _assert_shape(expected_shape, actual_shape):
    if not isinstance(expected_shape, (list, tuple, tensor_shape.TensorShape)):
        raise ValueError(
            'Wrong shape provided for "expected_shape"')
    if not isinstance(actual_shape, (list, tuple, tensor_shape.TensorShape)):
        raise ValueError(
            'Wrong shape provided for "actual_shape"')
    if len(expected_shape) != len(actual_shape):
        raise AssertionError('Wrong shape')

    for expected_dim, actual_dim in zip(expected_shape, actual_shape):
        if isinstance(expected_dim, tensor_shape.Dimension):
            expected_dim = expected_dim.value
        if isinstance(actual_dim, tensor_shape.Dimension):
            actual_dim = actual_dim.value
        if expected_dim is None:
            continue
        if expected_dim != actual_dim:
            raise AssertionError('Wrong shape')
