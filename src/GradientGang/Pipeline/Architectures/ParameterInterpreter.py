class ParameterInterpreter:
    def __init__(self, interpretation: dict, name: str = "ParameterInterpreter", requiredParams: dict = {}):
        self.interpretation = interpretation
        self.name = name
        self.requiredParams = requiredParams

    def checkRequiredParams(self, params: dict):
        for p in self.requiredParams:
            if p not in params:
                raise KeyError(
                    f"{self.name}: Required parameter '{p}' not found in provided parameters.")

            required_value = self.requiredParams[p]

            # Check if the required value is a type (class) for type checking
            if isinstance(required_value, type):
                if not isinstance(params.get(p), required_value):
                    raise TypeError(
                        f"{self.name}: Parameter '{p}' must be of type {required_value.__name__}.")
            # Recursively check nested required params if they exist (when it's a dict)
            elif isinstance(required_value, dict) and required_value:
                if not isinstance(params.get(p), dict):
                    raise TypeError(
                        f"{self.name}: Parameter '{p}' must be a dict for nested validation.")
                nested_interpreter = ParameterInterpreter(
                    interpretation={},
                    name=self.name,
                    requiredParams=required_value
                )
                nested_interpreter.checkRequiredParams(params[p])

    def interpret(self, param_name: str):
        if param_name in self.interpretation:
            return self.interpretation[param_name]
        else:
            raise KeyError(
                f"{self.name}: '{param_name}' not found in interpretation dictionary.")
