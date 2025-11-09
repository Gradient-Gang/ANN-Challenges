class ParameterInterpreter:
    def __init__(
        self, 
        interpretation: dict,
        name: str = "ParameterInterpreter", 
        requiredParams: dict = {}
    ):
        """
        The ParameterInterpreter class is responsible for interpreting parameters based on a provided interpretation dictionary.
        Args:
            interpretation (dict): A dictionary mapping parameter names to their corresponding classes or types.
            name (str): Name of the interpreter for identification in error
            requiredParams (dict): A dictionary specifying required parameters and their expected types or values.
        Raises:
            KeyError: If a required parameter is missing in the provided parameters.
            TypeError: If a required parameter does not match the expected type.
        """

        # Store the interpretation dictionary and other attributes
        self.interpretation = interpretation
        self.name = name
        self.requiredParams = requiredParams

    def checkRequiredParams(
        self, params: dict
    ):
        """
        Check if all required parameters are present and valid in the provided parameters.
        Args:
            params (dict): A dictionary of parameters to be checked.
        Raises: 
            KeyError: If a required parameter is missing in the provided parameters.
            TypeError: If a required parameter does not match the expected type.
        """

        # Iterate through each required parameter and validate
        for p in self.requiredParams:
            # Check if the required parameter is present
            if p not in params:
                raise KeyError(
                    f"{self.name}: Required parameter '{p}' not found in provided parameters.")

            # Retrieve the expected value or type for the required parameter
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

    def interpret(
        self, param_name: str
    ):
        """
        Interpret the given parameter name based on the interpretation dictionary.
        Args:
            param_name (str): The name of the parameter to interpret.
        Returns:
            The interpreted value or class associated with the parameter name.
        Raises:
            KeyError: If the parameter name is not found in the interpretation dictionary.
        """

        # Retrieve and return the interpreted value or class
        if param_name in self.interpretation:
            return self.interpretation[param_name]
        
        # Raise an error if the parameter name is not found
        else:
            raise KeyError(
                f"{self.name}: '{param_name}' not found in interpretation dictionary.")
