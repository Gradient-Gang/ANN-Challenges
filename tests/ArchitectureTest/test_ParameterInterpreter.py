from GradientGang.Pipeline.Architectures.ParameterInterpreter import ParameterInterpreter
import pytest


# Test cases for ParameterInterpreter initialization

def test_init_with_defaults():
    """Test initialization with default parameters"""
    interpretation = {"param1": "value1"}
    pi = ParameterInterpreter(interpretation)

    assert pi.interpretation == interpretation
    assert pi.name == "ParameterInterpreter"
    assert pi.requiredParams == {}


def test_init_with_custom_name():
    """Test initialization with custom name"""
    interpretation = {"param1": "value1"}
    custom_name = "CustomInterpreter"
    pi = ParameterInterpreter(interpretation, name=custom_name)

    assert pi.name == custom_name


def test_init_with_required_params():
    """Test initialization with required parameters"""
    interpretation = {"param1": "value1"}
    required_params = {"required1": {}, "required2": {}}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    assert pi.requiredParams == required_params


def test_init_with_all_params():
    """Test initialization with all parameters"""
    interpretation = {"param1": "value1", "param2": "value2"}
    name = "TestInterpreter"
    required_params = {"req1": {}, "req2": {}}

    pi = ParameterInterpreter(
        interpretation, name=name, requiredParams=required_params)

    assert pi.interpretation == interpretation
    assert pi.name == name
    assert pi.requiredParams == required_params


# Test cases for the interpret method

def test_interpret_existing_param():
    """Test interpreting an existing parameter"""
    interpretation = {"param1": "value1", "param2": 42, "param3": [1, 2, 3]}
    pi = ParameterInterpreter(interpretation)

    assert pi.interpret("param1") == "value1"
    assert pi.interpret("param2") == 42
    assert pi.interpret("param3") == [1, 2, 3]


def test_interpret_nonexistent_param():
    """Test interpreting a non-existent parameter raises KeyError"""
    interpretation = {"param1": "value1"}
    pi = ParameterInterpreter(interpretation)

    with pytest.raises(KeyError) as exc_info:
        pi.interpret("nonexistent")

    assert "nonexistent" in str(exc_info.value)
    assert "not found in interpretation dictionary" in str(exc_info.value)


def test_interpret_with_custom_name_in_error():
    """Test that custom name appears in error message"""
    interpretation = {"param1": "value1"}
    custom_name = "MyCustomInterpreter"
    pi = ParameterInterpreter(interpretation, name=custom_name)

    with pytest.raises(KeyError) as exc_info:
        pi.interpret("missing_param")

    assert custom_name in str(exc_info.value)


def test_interpret_none_value():
    """Test interpreting a parameter with None value"""
    interpretation = {"param_none": None}
    pi = ParameterInterpreter(interpretation)

    assert pi.interpret("param_none") is None


def test_interpret_empty_string():
    """Test interpreting a parameter with empty string value"""
    interpretation = {"empty": ""}
    pi = ParameterInterpreter(interpretation)

    assert pi.interpret("empty") == ""


def test_interpret_complex_objects():
    """Test interpreting parameters with complex object values"""
    interpretation = {
        "dict_param": {"nested": "value"},
        "list_param": [1, 2, {"key": "value"}],
        "tuple_param": (1, 2, 3)
    }
    pi = ParameterInterpreter(interpretation)

    assert pi.interpret("dict_param") == {"nested": "value"}
    assert pi.interpret("list_param") == [1, 2, {"key": "value"}]
    assert pi.interpret("tuple_param") == (1, 2, 3)


# Test cases for the checkRequiredParams method

def test_check_required_params_all_present():
    """Test checking required params when all are present"""
    interpretation = {}
    required_params = {"param1": {}, "param2": {}}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"param1": "value1", "param2": "value2"}

    # Should not raise an exception
    pi.checkRequiredParams(params)


def test_check_required_params_missing_param():
    """Test checking required params when one is missing"""
    interpretation = {}
    required_params = {"param1": {}, "param2": {}}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"param1": "value1"}  # param2 is missing

    with pytest.raises(KeyError) as exc_info:
        pi.checkRequiredParams(params)

    assert "param2" in str(exc_info.value)
    assert "Required parameter" in str(exc_info.value)


def test_check_required_params_custom_name_in_error():
    """Test that custom name appears in error message"""
    interpretation = {}
    required_params = {"required_field": {}}
    custom_name = "ValidationInterpreter"
    pi = ParameterInterpreter(
        interpretation, name=custom_name, requiredParams=required_params)

    params = {}

    with pytest.raises(KeyError) as exc_info:
        pi.checkRequiredParams(params)

    assert custom_name in str(exc_info.value)


def test_check_required_params_empty_required():
    """Test checking when no required params are defined"""
    interpretation = {}
    pi = ParameterInterpreter(interpretation)

    params = {"any": "value"}
    # Should not raise an exception
    pi.checkRequiredParams(params)


def test_check_required_params_empty_params_empty_required():
    """Test with both empty params and empty required"""
    interpretation = {}
    pi = ParameterInterpreter(interpretation, requiredParams={})

    params = {}
    # Should not raise exception when nothing is required
    pi.checkRequiredParams(params)


def test_check_required_params_nested_required():
    """Test checking with nested required parameters"""
    interpretation = {}
    required_params = {
        "level1": {
            "level2": {}
        }
    }
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    # Test with nested param present
    params = {"level1": {"level2": "value"}}

    # Should not raise exception when nested params are present
    pi.checkRequiredParams(params)


def test_check_required_params_nested_missing():
    """Test checking with nested required parameters when nested is missing"""
    interpretation = {}
    required_params = {
        "level1": {
            "level2": {}
        }
    }
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    # Test with level1 present but level2 missing
    params = {"level1": {}}

    with pytest.raises(KeyError) as exc_info:
        pi.checkRequiredParams(params)

    assert "level2" in str(exc_info.value)


def test_check_required_params_with_none_value():
    """Test checking required params when param is present but None"""
    interpretation = {}
    required_params = {"param1": {}}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"param1": None}
    # Should not raise an exception - the param key exists
    pi.checkRequiredParams(params)


def test_check_required_params_multiple_missing():
    """Test checking when multiple required params are missing"""
    interpretation = {}
    required_params = {"param1": {}, "param2": {}, "param3": {}}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {}

    # Should raise KeyError for the first missing param encountered
    with pytest.raises(KeyError):
        pi.checkRequiredParams(params)


def test_check_required_params_signature():
    """Test that checkRequiredParams method has correct signature"""
    interpretation = {}
    pi = ParameterInterpreter(interpretation)

    # Verify method accepts exactly one parameter (params)
    import inspect
    sig = inspect.signature(pi.checkRequiredParams)
    params = list(sig.parameters.keys())

    assert len(
        params) == 1, "checkRequiredParams should accept exactly 1 parameter"
    assert params[0] == "params", "Parameter should be named 'params'"


def test_check_required_params_with_type_int():
    """Test checking required params with int type"""
    interpretation = {}
    required_params = {"param1": int, "param2": str}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"param1": 42, "param2": "hello"}

    # Should not raise an exception
    pi.checkRequiredParams(params)


def test_check_required_params_with_type_mismatch():
    """Test checking required params with type mismatch"""
    interpretation = {}
    required_params = {"param1": int}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"param1": "not an int"}

    with pytest.raises(TypeError) as exc_info:
        pi.checkRequiredParams(params)

    assert "param1" in str(exc_info.value)
    assert "must be of type int" in str(exc_info.value)


def test_check_required_params_with_type_list():
    """Test checking required params with list type"""
    interpretation = {}
    required_params = {"items": list, "count": int}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"items": [1, 2, 3], "count": 3}

    # Should not raise an exception
    pi.checkRequiredParams(params)


def test_check_required_params_with_type_dict():
    """Test checking required params with dict type"""
    interpretation = {}
    required_params = {"config": dict}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"config": {"key": "value"}}

    # Should not raise an exception
    pi.checkRequiredParams(params)


def test_check_required_params_nested_with_types():
    """Test checking nested required params with type checking"""
    interpretation = {}
    required_params = {
        "model": {
            "name": str,
            "layers": int
        }
    }
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {
        "model": {
            "name": "ResNet",
            "layers": 50
        }
    }

    # Should not raise an exception
    pi.checkRequiredParams(params)


def test_check_required_params_nested_with_type_mismatch():
    """Test checking nested required params with type mismatch"""
    interpretation = {}
    required_params = {
        "model": {
            "name": str,
            "layers": int
        }
    }
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {
        "model": {
            "name": "ResNet",
            "layers": "fifty"  # Should be int
        }
    }

    with pytest.raises(TypeError) as exc_info:
        pi.checkRequiredParams(params)

    assert "layers" in str(exc_info.value)
    assert "must be of type int" in str(exc_info.value)


def test_check_required_params_nested_not_dict():
    """Test that nested validation requires param to be a dict"""
    interpretation = {}
    required_params = {
        "model": {
            "name": str
        }
    }
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"model": "not a dict"}

    with pytest.raises(TypeError) as exc_info:
        pi.checkRequiredParams(params)

    assert "model" in str(exc_info.value)
    assert "must be a dict" in str(exc_info.value)


def test_check_required_params_mixed_types_and_nested():
    """Test mixing type requirements and nested dict requirements"""
    interpretation = {}
    required_params = {
        "name": str,
        "age": int,
        "config": {
            "verbose": bool,
            "timeout": int
        }
    }
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {
        "name": "Alice",
        "age": 30,
        "config": {
            "verbose": True,
            "timeout": 60
        }
    }

    # Should not raise an exception
    pi.checkRequiredParams(params)


def test_check_required_params_with_none_type():
    """Test that None values fail type checking"""
    interpretation = {}
    required_params = {"param1": str}
    pi = ParameterInterpreter(interpretation, requiredParams=required_params)

    params = {"param1": None}

    with pytest.raises(TypeError) as exc_info:
        pi.checkRequiredParams(params)

    assert "param1" in str(exc_info.value)
    assert "must be of type str" in str(exc_info.value)


# Integration tests for ParameterInterpreter

def test_full_workflow():
    """Test a complete workflow with interpretation and validation"""
    interpretation = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam"
    }
    required_params = {
        "learning_rate": {},
        "batch_size": {}
    }

    pi = ParameterInterpreter(
        interpretation,
        name="ModelConfig",
        requiredParams=required_params
    )

    # Validate params
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
    pi.checkRequiredParams(params)

    # Interpret values
    lr = pi.interpret("learning_rate")
    batch = pi.interpret("batch_size")
    opt = pi.interpret("optimizer")

    assert lr == 0.001
    assert batch == 32
    assert opt == "adam"


def test_full_workflow_with_nested_validation():
    """Test workflow with nested parameter validation"""
    interpretation = {
        "model": {
            "architecture": "ResNet",
            "layers": 50
        }
    }
    required_params = {
        "model": {
            "architecture": {},
            "layers": {}
        }
    }

    pi = ParameterInterpreter(
        interpretation,
        name="NestedConfig",
        requiredParams=required_params
    )

    # Validate nested params
    params = {
        "model": {
            "architecture": "ResNet",
            "layers": 50
        }
    }
    pi.checkRequiredParams(params)

    # Interpret values
    model_config = pi.interpret("model")
    assert model_config["architecture"] == "ResNet"
    assert model_config["layers"] == 50


def test_edge_case_empty_interpretation():
    """Test with empty interpretation dictionary"""
    pi = ParameterInterpreter({})

    with pytest.raises(KeyError):
        pi.interpret("any_param")


def test_edge_case_special_characters_in_keys():
    """Test with special characters in parameter names"""
    interpretation = {
        "param-with-dash": "value1",
        "param_with_underscore": "value2",
        "param.with.dot": "value3"
    }
    pi = ParameterInterpreter(interpretation)

    assert pi.interpret("param-with-dash") == "value1"
    assert pi.interpret("param_with_underscore") == "value2"
    assert pi.interpret("param.with.dot") == "value3"
